import io
import json
from email.message import Message
from http.client import IncompleteRead, RemoteDisconnected
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from volleymole import llm_transport as transport


class LLMTransportTests(unittest.TestCase):
    payload = {'model': 'glm-5.3-flash', 'messages': [], 'response_format': {'type': 'json_object'}}

    def response(self, content='{"events":[]}', finish='stop', **extra):
        data = {'choices': [{'finish_reason': finish, 'message': {'content': content}}], **extra}
        response = io.BytesIO(json.dumps(data).encode())
        response.headers = Message()
        return response

    def request(self, response):
        opener = Mock(return_value=response)
        return transport.request_json('https://example.invalid/v1', 'PRIVATE_KEY', self.payload, 5, opener=opener)

    def test_plain_and_whole_fenced_json_are_accepted(self):
        for content, encoding in (
                ('  {"events":[]} \n', 'json'),
                ('```json\n{"events":[]}\n```', 'fenced_json'),
                ('```JSON\r\n{"events":[]}\r\n```', 'fenced_json'),
                ('```\n{"events":[]}\n```', 'fenced_json')):
            with self.subTest(content=content):
                data, metadata = self.request(self.response(content))
                self.assertEqual(data, {'events': []})
                self.assertEqual(metadata['response_content_encoding'], encoding)

    def test_mixed_prose_truncation_and_nonstandard_json_stay_failures(self):
        for content in ('Here is JSON: {"events":[]}', '{"events":[]} trailing',
                        '```json\n{"events":[]}\n```\nDone', '```json\n{"events":[]}',
                        '```python\n{"events":[]}\n```', '{"events":[],}',
                        '{"events":[]} {"events":[]}', '{"events":[],"events":[1]}',
                        '{"score":NaN}', '{"score":Infinity}', '{"score":-Infinity}', '{"score":1e999}'):
            with self.subTest(content=content), self.assertRaises(transport.ResponseContractError) as raised:
                self.request(self.response(content))
            self.assertEqual(raised.exception.reason, 'invalid_json')

    def test_transport_does_not_coerce_valid_but_wrong_domain_shape(self):
        for value in ([{'events': []}], {'wrong': []}, {'events': [7]}, None):
            with self.subTest(value=value):
                parsed, _ = self.request(self.response(json.dumps(value)))
                self.assertEqual(parsed, value)

    def test_bad_envelope_is_safe_and_classified(self):
        for envelope in (None, [], {}, {'choices': []}, {'choices': [None]},
                         {'choices': [{'message': None}]}, {'choices': [{}, {}]},
                         {'choices': [{'finish_reason': 'stop', 'message': {}}]}):
            response = io.BytesIO(json.dumps(envelope).encode())
            with self.subTest(envelope=envelope), self.assertRaises(transport.ResponseContractError) as raised:
                self.request(response)
            self.assertEqual(raised.exception.reason, 'invalid_response_envelope')
            self.assertNotIn('PRIVATE_KEY', str(raised.exception))
        with self.assertRaises(transport.ResponseContractError) as raised:
            self.request(io.BytesIO(b'PRIVATE_RESPONSE_NOT_JSON'))
        self.assertNotIn('PRIVATE_RESPONSE', str(raised.exception))

    def test_finish_and_refusal_never_get_overridden_by_valid_json(self):
        for finish in ('length', 'content_filter', 'tool_calls', 'PRIVATE_REASON', None):
            with self.subTest(finish=finish), self.assertRaises(transport.ResponseContractError) as raised:
                self.request(self.response(finish=finish))
            self.assertEqual(raised.exception.reason, 'incomplete_response')
            self.assertNotIn('PRIVATE_REASON', str(vars(raised.exception)))
        response = io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
            'message': {'content': '{"events":[]}', 'refusal': 'PRIVATE_REFUSAL'}}]}).encode())
        with self.assertRaises(transport.ResponseContractError) as raised:
            self.request(response)
        self.assertEqual(raised.exception.reason, 'refused')
        self.assertNotIn('PRIVATE_REFUSAL', str(vars(raised.exception)))

    def test_disconnect_and_truncated_http_body_are_retryable_and_redacted(self):
        for error in (RemoteDisconnected('PRIVATE_RESPONSE'), IncompleteRead(b'PRIVATE_RESPONSE', 42),
                      ConnectionResetError('PRIVATE_KEY'), BrokenPipeError('PRIVATE_KEY'),
                      TimeoutError('PRIVATE_KEY'), URLError('PRIVATE_KEY')):
            opener = Mock(side_effect=error)
            with self.subTest(error=type(error).__name__), self.assertRaises(transport.TransportError) as raised:
                transport.request_json('https://example.invalid', 'PRIVATE_KEY', self.payload, 5, opener=opener)
            self.assertTrue(transport.is_retryable_transport_error(raised.exception))
            self.assertTrue(transport.is_retryable_transport_error(error))
            self.assertNotIn('PRIVATE_', str(raised.exception))
            self.assertEqual(opener.call_count, 1, 'Transport must not add a nested retry budget')

    def test_body_disconnect_also_closes_response_and_is_retryable(self):
        response = self.response()
        response.read1 = Mock(side_effect=IncompleteRead(b'PRIVATE_RESPONSE', 99))
        with self.assertRaises(transport.TransportError) as raised:
            self.request(response)
        self.assertTrue(response.closed)
        self.assertTrue(transport.is_retryable_transport_error(raised.exception))
        self.assertNotIn('PRIVATE_', str(raised.exception))

    def test_http_error_policy_and_body_cleanup(self):
        for status, retryable in ((400, False), (401, False), (403, False), (408, True),
                                  (422, False), (429, True), (500, True), (503, True)):
            body = io.BytesIO(b'PRIVATE_RESPONSE')
            error = HTTPError('https://example.invalid/PRIVATE_KEY', status, 'PRIVATE_RESPONSE', {}, body)
            with self.subTest(status=status), self.assertRaises(transport.TransportError) as raised:
                transport.request_json('https://example.invalid', 'PRIVATE_KEY', self.payload, 5,
                                       opener=Mock(side_effect=error))
            self.assertEqual(raised.exception.http_status, status)
            self.assertEqual(transport.is_retryable_transport_error(raised.exception), retryable)
            self.assertEqual(transport.is_retryable_transport_error(error), retryable)
            self.assertTrue(body.closed)
            self.assertNotIn('PRIVATE_', str(raised.exception))

    def test_legacy_http_error_keeps_identity_and_body_for_caller(self):
        body = io.BytesIO(b'{"error":{"code":"legacy_classifier"}}')
        error = HTTPError('https://example.invalid', 400, 'Bad Request', {}, body)
        with self.assertRaises(HTTPError) as raised:
            transport.request_json('https://example.invalid', 'PRIVATE_KEY', self.payload, 5,
                                   opener=Mock(side_effect=error), preserve_http_error=True)
        self.assertIs(raised.exception, error)
        self.assertFalse(body.closed)
        self.assertEqual(json.load(raised.exception)['error']['code'], 'legacy_classifier')
        raised.exception.close()
        self.assertTrue(body.closed)

    def test_nontransient_io_errors_are_not_retried(self):
        with self.assertRaises(transport.TransportError) as raised:
            transport.request_json('https://example.invalid', 'PRIVATE_KEY', self.payload, 5,
                                   opener=Mock(side_effect=OSError('PRIVATE_KEY')))
        self.assertFalse(transport.is_retryable_transport_error(raised.exception))
        self.assertFalse(transport.is_retryable_transport_error(transport.ResponseContractError('invalid_json')))

    def test_envelope_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        for raw in (b'{"choices":[],"choices":[]}', b'{"choices":[],"usage":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(transport.ResponseContractError) as raised:
                self.request(io.BytesIO(raw))
            self.assertEqual(raised.exception.reason, 'invalid_response_envelope')

    def test_size_limit_and_absolute_deadline_cover_reads_and_open(self):
        with patch.object(transport, 'MAX_RESPONSE_BYTES', 8), self.assertRaises(transport.ResponseContractError) as raised:
            self.request(self.response())
        self.assertEqual(raised.exception.reason, 'response_too_large')
        response = self.response()
        with patch.object(transport.time, 'monotonic', side_effect=[0, 1, 6]), \
             self.assertRaises(transport.TransportError) as raised:
            self.request(response)
        self.assertEqual(raised.exception.reason, 'timeout')
        self.assertTrue(response.closed)
        with patch.object(transport.time, 'monotonic', side_effect=[0, 1, 2, 6]), \
             self.assertRaises(transport.TransportError) as raised:
            self.request(self.response())
        self.assertEqual(raised.exception.reason, 'timeout')

    def test_open_and_socket_receive_only_remaining_deadline(self):
        response = self.response()
        socket = Mock()
        response.fp = Mock(raw=Mock(_sock=socket))
        opener = Mock(return_value=response)
        # Opening receives 4 s; first body read receives 3 s, EOF read 2 s.
        with patch.object(transport.time, 'monotonic', side_effect=[0, 1, 2, 2.5, 3, 3.5]):
            data, _ = transport.request_json('https://example.invalid', 'PRIVATE_KEY',
                                             self.payload, 5, opener=opener)
        self.assertEqual(data, {'events': []})
        self.assertEqual(opener.call_args.kwargs['timeout'], 4)
        self.assertEqual([call.args[0] for call in socket.settimeout.call_args_list], [3, 2])
        self.assertTrue(response.closed)

    def test_unknown_gateway_metadata_and_usage_strings_are_not_persisted(self):
        response = self.response(usage={'prompt_tokens': 123, 'completion_tokens': 5,
            'total_tokens': 'PRIVATE_USAGE', 'untrusted': 'PRIVATE_USAGE',
            'prompt_tokens_details': {'cached_tokens': 100, 'untrusted': 'PRIVATE_USAGE'},
            'completion_tokens_details': {'reasoning_tokens': -5, 'audio_tokens': True}})
        response.headers['X-Structured-Output-Degraded'] = 'PRIVATE_HEADER'
        _, metadata = self.request(response)
        self.assertEqual(metadata['structured_output_degraded'], 'other')
        self.assertEqual(metadata['usage'], {'prompt_tokens': 123, 'completion_tokens': 5,
            'prompt_tokens_details': {'cached_tokens': 100}, 'completion_tokens_details': {}})
        self.assertNotIn('PRIVATE_', json.dumps(metadata))

    def test_gateway_repair_header_is_boolean_and_does_not_bypass_validation(self):
        for header, expected in ((None, False), ('true', True), ('false', False), ('PRIVATE_HEADER', False)):
            response = self.response()
            if header is not None:
                response.headers['X-JSON-Repaired'] = header
            with self.subTest(header=header):
                _, metadata = self.request(response)
                self.assertIs(metadata['gateway_json_repaired'], expected)
                self.assertNotIn('PRIVATE_', json.dumps(metadata))
        for content, finish, reason in (('{"events":[]}', 'length', 'incomplete_response'),
                                        ('{"events":[],}', 'stop', 'invalid_json')):
            response = self.response(content, finish)
            response.headers['X-JSON-Repaired'] = 'true'
            with self.subTest(reason=reason), self.assertRaises(transport.ResponseContractError) as raised:
                self.request(response)
            self.assertEqual(raised.exception.reason, reason)

    def test_invalid_timeout_is_rejected_before_network(self):
        opener = Mock()
        for timeout in (0, -1, float('inf'), float('nan'), True, '5'):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                transport.request_json('https://example.invalid', 'PRIVATE_KEY', self.payload, timeout, opener=opener)
        opener.assert_not_called()


if __name__ == '__main__':
    unittest.main()
