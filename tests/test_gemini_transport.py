import io
import json
from http.client import IncompleteRead, RemoteDisconnected
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

from volleymole.gemini_transport import request_json
from volleymole.llm_transport import ResponseContractError, TransportError
from volleymole.replay_requests import _safe_request_metadata


class ChunkedResponse:
    headers = {'Content-Type':'text/event-stream; charset=utf-8'}
    status = 200
    def __init__(self,chunks):self.chunks=iter(chunks);self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def read1(self,size):
        value=next(self.chunks,b'')
        if isinstance(value,BaseException):raise value
        return value


class GeminiTransportTests(unittest.TestCase):
    def wire(self):
        return {'model':'gemini-3.1-pro-preview','max_tokens':1024,'reasoning_effort':'low',
            'messages':[{'role':'system','content':'Return JSON'},{'role':'user','content':[{'type':'text','text':'ok'}]}],
            'response_format':{'json_schema':{'schema':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']}}}}

    def event(self,text='{"ok":true}',**extra):
        return ('data: '+json.dumps({'candidates':[{'index':0,'content':{'role':'model','parts':[{'text':text}]},
                                  'finishReason':'STOP'}],**extra})+'\n\n').encode()

    def call(self,response=None,opener=None,progress=None):
        return request_json('https://aihubmix.com/v1','PRIVATE_KEY',self.wire(),5,
                            opener=opener or Mock(return_value=response),progress=progress)

    def test_bom_split_crlf_unicode_and_thoughts_do_not_leak(self):
        first='data: '+json.dumps({'candidates':[{'content':{'parts':[{'thought':True,'text':'PRIVATE_THOUGHT'}]}}]})+'\r\n\r\n'
        raw=b'\xef\xbb\xbf: heartbeat\r\n\r\n'+first.encode()+self.event('{"ok":true}').replace(b'\n',b'\r\n')
        response=ChunkedResponse([raw[i:i+7] for i in range(0,len(raw),7)])
        progress=[];result,metadata=self.call(response,progress=progress.append)
        self.assertEqual(result,{'ok':True});self.assertTrue(response.closed)
        self.assertEqual(metadata['transport']['heartbeat_count'],1)
        self.assertEqual(metadata['transport']['reasoning_delta_count'],1)
        self.assertTrue(metadata['transport']['clean_eof'])
        self.assertNotIn('PRIVATE_',json.dumps([metadata,progress]))

    def test_underlying_open_error_and_mid_stream_disconnect_are_distinct(self):
        for error in (URLError(ConnectionRefusedError(111,'PRIVATE_KEY')),RemoteDisconnected('PRIVATE_KEY')):
            with self.subTest(error=type(error).__name__),self.assertRaises(TransportError) as raised:
                self.call(opener=Mock(side_effect=error))
            details=raised.exception.diagnostics
            self.assertEqual(details['phase'],'open');self.assertEqual(details['received_bytes'],0)
            self.assertIn(details['cause_class'],('URLError','RemoteDisconnected'))
            if isinstance(error,URLError):
                self.assertEqual(details['underlying_cause_class'],'ConnectionRefusedError')
                self.assertEqual(details['underlying_errno'],111)
            self.assertNotIn('PRIVATE_',json.dumps(details))
        response=ChunkedResponse([b': heartbeat\n\n',IncompleteRead(b'PRIVATE_BODY',42)])
        with self.assertRaises(TransportError) as raised:self.call(response)
        details=raised.exception.diagnostics
        self.assertEqual(details['phase'],'read');self.assertEqual(details['cause_class'],'IncompleteRead')
        self.assertEqual(details['received_bytes'],13);self.assertTrue(response.closed)

    def test_provider_http_error_is_preserved_without_body_or_key_echo(self):
        body=io.BytesIO(json.dumps({'error':{'code':'upstream_error','type':'server_error','param':'PRIVATE_KEY','message':'PRIVATE_BODY'}}).encode())
        error=HTTPError('https://aihubmix.com/private',502,'PRIVATE_KEY',{},body)
        with self.assertRaises(TransportError) as raised:self.call(opener=Mock(side_effect=error))
        self.assertEqual(raised.exception.http_status,502)
        details=raised.exception.diagnostics
        self.assertEqual(details['http_error_body']['code'],'upstream_error')
        self.assertNotIn('PRIVATE_',json.dumps(details));self.assertTrue(body.closed)

    def test_usage_survives_the_existing_audit_filter(self):
        response=ChunkedResponse([self.event(usageMetadata={'promptTokenCount':1000,'candidatesTokenCount':50,
            'thoughtsTokenCount':100,'totalTokenCount':1150,'promptTokensDetails':[{'modality':'IMAGE','tokenCount':900}]},
            modelVersion='gemini-3.1-pro-preview',responseId='safe-request-01')])
        _,metadata=self.call(response)
        saved=_safe_request_metadata(metadata)
        self.assertEqual(saved['usage']['prompt_tokens_details']['image_tokens'],900)
        self.assertEqual(saved['usage']['completion_tokens'],150)
        self.assertEqual(metadata['transport']['model_version'],'gemini-3.1-pro-preview')

    def test_final_json_failure_retains_only_final_answer_for_redacted_diagnostics(self):
        response=ChunkedResponse([self.event('not JSON')])
        with self.assertRaises(ResponseContractError) as raised:self.call(response)
        self.assertEqual(raised.exception.final_text,'not JSON')
        self.assertEqual(raised.exception.diagnostics['phase'],'parse')

    def test_incomplete_event_after_stop_is_rejected(self):
        response=ChunkedResponse([self.event(),b'data: {"error":'])
        with self.assertRaises(ResponseContractError):self.call(response)

    def test_exhausted_output_budget_preserves_usage_and_incomplete_final_only(self):
        chunks=[{'candidates':[{'content':{'parts':[{'thought':True,'text':'PRIVATE_THOUGHT'}]}}]},
                {'candidates':[{'content':{'parts':[{'text':'{"ok":'}]},'finishReason':'MAX_TOKENS'}],
                 'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':24,'thoughtsTokenCount':1000,'totalTokenCount':1124}}]
        response=ChunkedResponse([('data: '+json.dumps(c)+'\n\n').encode() for c in chunks])
        with self.assertRaises(ResponseContractError) as raised:self.call(response)
        error=raised.exception
        self.assertEqual(error.finish_reason,'length')
        self.assertEqual(error.final_text,'{"ok":')
        self.assertEqual(error.diagnostics['usage']['completion_tokens'],1024)
        self.assertNotIn('PRIVATE_THOUGHT',json.dumps(error.diagnostics)+error.final_text)


if __name__=='__main__':unittest.main()
