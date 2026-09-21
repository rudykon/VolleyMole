"""Native Gemini SSE using the replay-tested parser; no retries or redirects."""
import base64
import io
import json
import math
import re
import time
from http.client import HTTPException
from urllib.error import HTTPError,URLError
from urllib.parse import urlsplit
from urllib.request import Request,build_opener,HTTPRedirectHandler
from . import gemini_stream as shared
from .llm_transport import _strict_json,parse_json_content,ResponseContractError,TransportError
ENDPOINT=shared.ENDPOINT
SSEError=shared.SSEError
class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def native_payload(endpoint, payload, timeout):
    parts = urlsplit(endpoint)
    model = payload.get('model', '')
    if (parts.scheme != 'https' or not parts.netloc or parts.username or parts.password
            or parts.query or parts.fragment or parts.path.rstrip('/') != '/v1'
            or not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9._-]+', model)
            or type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Invalid native endpoint, model or timeout')
    system, user = payload['messages']
    if system['role'] != 'system' or user['role'] != 'user':
        raise ValueError('Expected system and user messages')
    content = []
    for item in user['content']:
        if item['type'] == 'text':
            content.append({'text': item['text']})
        elif item['type'] == 'image_url':
            url = item['image_url']['url']
            prefix = 'data:image/jpeg;base64,'
            if not url.startswith(prefix):
                raise ValueError('Only inline JPEG frames are supported')
            base64.b64decode(url[len(prefix):], validate=True)
            content.append({'inlineData': {'mimeType': 'image/jpeg', 'data': url[len(prefix):]}})
        else:
            raise ValueError('Unsupported native content type')
    if payload.get('reasoning_effort','high') not in ('low','medium','high'):
        raise ValueError('Unsupported Gemini thinking level')
    body = {'systemInstruction': {'parts': [{'text': system['content']}]},
            'contents': [{'role': 'user', 'parts': content}],
            'generationConfig': {'maxOutputTokens': payload['max_tokens'],
                # Match the previously successful streaming configuration.
                # Stream thought chunks to avoid a long silent connection;
                # decode_stream discards them and retains only the final JSON.
                'thinkingConfig': {'thinkingLevel': payload.get('reasoning_effort', 'high').upper(), 'includeThoughts': True},
                'mediaResolution': 'MEDIA_RESOLUTION_HIGH', 'responseMimeType': 'application/json',
                'responseJsonSchema': payload['response_format']['json_schema']['schema']}}
    return body


def safe_usage(value):
    if not isinstance(value,dict):return None
    mapping={'promptTokenCount':'prompt_tokens','candidatesTokenCount':'visible_completion_tokens',
             'thoughtsTokenCount':'reasoning_tokens','totalTokenCount':'total_tokens','cachedContentTokenCount':'cached_tokens'}
    counters={out:value[k] for k,out in mapping.items() if type(value.get(k)) is int and value[k]>=0}
    result={k:counters[k] for k in ('prompt_tokens','total_tokens') if k in counters}
    if 'visible_completion_tokens' in counters:
        result['completion_tokens']=counters['visible_completion_tokens']+counters.get('reasoning_tokens',0)
    result['completion_tokens_details']={'reasoning_tokens':counters['reasoning_tokens']} if 'reasoning_tokens' in counters else {}
    details={'cached_tokens':counters['cached_tokens']} if 'cached_tokens' in counters else {}
    for item in value.get('promptTokensDetails',[]) if isinstance(value.get('promptTokensDetails',[]),list) else []:
        if isinstance(item,dict) and item.get('modality')=='IMAGE' and type(item.get('tokenCount')) is int and item['tokenCount']>=0:
            details['image_tokens']=details.get('image_tokens',0)+item['tokenCount']
    result['prompt_tokens_details']=details
    return result

def _request_native(endpoint,key,payload,timeout,*,opener=None,clock=time.monotonic,url=None,progress=None):
    if endpoint.rstrip('/')!=ENDPOINT or type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('Native endpoint or deadline outside scope')
    state=dict(began=clock(),received_bytes=0,event_count=0,heartbeat_count=0,content_bytes=0,reasoning_delta_count=0,phase='open')
    state['_progress']=progress
    deadline=state['began']+timeout
    req=Request(url,data=json.dumps(payload,allow_nan=False).encode(),
                headers={'x-goog-api-key':key,'Content-Type':'application/json','Accept':'text/event-stream'})
    state.update(request_bytes=len(req.data),image_count=sum('inlineData' in p for p in payload['contents'][0]['parts']))
    if progress:progress(shared._audit(state,clock))
    limits=dict(wire=shared.MAX_WIRE_BYTES,event=shared.MAX_EVENT_BYTES,content=shared.MAX_CONTENT_BYTES,events=shared.MAX_EVENTS)
    error_result=None;answer=None;metadata=None;content=[];usage=None
    fail=lambda reason,**extra:shared._fail(reason,state,clock,**extra)
    try:
        with (opener or build_opener(shared.NoRedirect()).open)(req,timeout=shared._remaining(state,deadline,clock)) as response:
            shared._remaining(state,deadline,clock)
            state.update(phase='read',headers_received_sec=round(clock()-state['began'],6),http_status=getattr(response,'status',200))
            if progress:progress(shared._audit(state,clock))
            if str(response.headers.get('Content-Type','')).split(';',1)[0].strip().lower()!='text/event-stream':
                fail('invalid_content_type')
            finish=None
            for event,data in shared._events(response,state,deadline,clock,limits):
                if event not in (b'',b'message',b'error'):fail('invalid_stream_event')
                if event==b'error':fail('provider_stream_error')
                if data.strip()==b'[DONE]':
                    if finish!='STOP':fail('incomplete_response')
                    state['done_received']=True
                    continue
                if state.get('done_received'):fail('content_after_finish')
                chunk=_strict_json(data,'invalid_stream_event')
                if not isinstance(chunk,dict):fail('invalid_stream_event')
                if 'error' in chunk:fail('provider_stream_error')
                for original,name in [('modelVersion','model_version'),('responseId','response_id')]:
                    value=chunk.get(original)
                    if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}',value) and not (key and key in value):state[name]=value
                if not isinstance(chunk.get('promptFeedback',{}),dict):fail('invalid_stream_field')
                if chunk.get('promptFeedback',{}).get('blockReason'):fail('refused')
                if chunk.get('usageMetadata') is not None:usage=safe_usage(chunk['usageMetadata'])
                candidates=chunk.get('candidates',[])
                if not candidates and chunk.get('usageMetadata') is not None:continue
                if not isinstance(candidates,list) or len(candidates)!=1:fail('unsupported_choice')
                candidate=candidates[0]
                if not isinstance(candidate,dict) or candidate.get('index',0)!=0:fail('unsupported_choice')
                body=candidate.get('content',{})
                if not isinstance(body,dict) or body.get('role','model')!='model' or not isinstance(body.get('parts',[]),list):fail('invalid_stream_field')
                for part in body.get('parts',[]):
                    if not isinstance(part,dict):fail('invalid_stream_field')
                    if part.get('thought') is True:
                        state['reasoning_delta_count']+=1;continue
                    if set(part)-{'text','thought','thoughtSignature'}:fail('invalid_stream_field')
                    piece=part.get('text','')
                    if not isinstance(piece,str):fail('invalid_stream_field')
                    if piece:
                        if finish is not None:fail('content_after_finish')
                        state.setdefault('first_content_sec',round(clock()-state['began'],6))
                        state['content_bytes']+=len(piece.encode())
                        if state['content_bytes']>limits['content']:fail('response_too_large')
                        content.append(piece)
                reason=candidate.get('finishReason')
                if reason is not None:
                    if reason!='STOP':fail('incomplete_response',finish_reason='length' if reason=='MAX_TOKENS' else 'other')
                    if finish is not None:fail('invalid_stream_event')
                    finish=reason
            if finish!='STOP':fail('incomplete_response')
            shared._remaining(state,deadline,clock);state['phase']='parse'
            answer,encoding=parse_json_content(''.join(content),'stop')
            metadata=dict(finish_reason='stop',response_format='json_schema',reasoning_effort=payload.get('generationConfig',{}).get('thinkingConfig',{}).get('thinkingLevel','HIGH').lower(),
                response_content_encoding=encoding,gateway_json_repaired=False,usage=usage,
                stream_audit=shared._audit(state,clock))
    except SSEError as error:error_result=error
    except HTTPError as error:
        remaining=deadline-clock()
        if remaining>0:
            sock=getattr(getattr(getattr(error,'fp',None),'raw',None),'_sock',None)
            if sock is not None:sock.settimeout(remaining)
            safe=shared.safe_http_body(error,key)
        else:error.close();safe={'body_available':False,'body_read_skipped':'deadline'}
        error_result=SSEError('http_error',{**shared._audit(state,clock),'http_status':error.code,'http_error_body':safe})
    except TimeoutError as error:error_result=SSEError('timeout',{**shared._audit(state,clock),**shared._cause(error)})
    except (URLError,ConnectionError,HTTPException,OSError) as error:
        error_result=SSEError('connection_error',{**shared._audit(state,clock),**shared._cause(error)})
    except ResponseContractError as error:
        error_result=SSEError('invalid_json',{**shared._audit(state,clock),'contract_error':error.reason})
        if state['phase']=='parse':error_result.final_text=''.join(content)
    if error_result is not None:
        # Usage and an unfinished final answer matter for diagnosing MAX_TOKENS.
        # Thought parts never enter content and must not enter failure artifacts.
        if usage is not None:error_result.diagnostics['usage']=usage
        if content:error_result.final_text=''.join(content)
        error_result.__context__=None;error_result.__cause__=None
        raise error_result from None
    return answer,metadata

def request_json(endpoint,key,payload,timeout,*,opener=None,clock=time.monotonic,progress=None):
    body=native_payload(endpoint,payload,timeout)
    parts=urlsplit(endpoint)
    url=f'{parts.scheme}://{parts.netloc}/gemini/v1beta/models/{payload["model"]}:streamGenerateContent?alt=sse'
    try:
        answer,metadata=_request_native(endpoint,key,body,timeout,opener=opener,clock=clock,url=url,progress=progress)
    except SSEError as error:
        if error.reason in ('http_error','timeout','connection_error','io_error'):
            status=error.diagnostics.get('http_status')
            retryable=error.reason in ('timeout','connection_error') or status in (408,429) or isinstance(status,int) and status>=500
            failure=TransportError(error.reason,retryable=retryable,http_status=status)
        else:
            failure=ResponseContractError(error.reason,error.diagnostics.get('finish_reason'))
        failure.diagnostics=error.diagnostics
        if hasattr(error,'final_text'):failure.final_text=error.final_text
        raise failure from None
    metadata['transport']=metadata.pop('stream_audit')
    return answer,metadata


def decode_stream(raw):
    response=io.BytesIO(raw)
    response.headers={'Content-Type':'text/event-stream'}
    body={'contents':[{'parts':[]}]}
    try:
        return _request_native(ENDPOINT,'',body,10,opener=lambda *a,**kw:response,url='https://example.invalid')
    except SSEError as error:
        raise ResponseContractError(error.reason) from None
