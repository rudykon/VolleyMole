"""Recovery is bounded and auditable, and never turns invalid events into empties."""
import copy
import io
import json
import tempfile
import time
import unittest
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from volleymole import semantic
from volleymole.event_schema import decode_event_response
from test_event_wire import evidence, wire_event


class SemanticRetryTests(unittest.TestCase):
    def settings(self, retries=1):
        return {'model': 'glm-5.3-flash', 'endpoint': 'https://example.invalid/v1',
                'key': 'PRIVATE_API_KEY', 'retries': retries, 'timeout': 10,
                'coarse_fps': 2, 'review_fps': 8, 'modality': 'frames',
                'frame_width': 224, 'max_tokens': 4096, 'reasoning_effort': 'low'}

    def sampler(self, *args, **kwargs):
        refs = evidence()
        frames = [(ref['start_sec'], np.zeros((2, 2), dtype=np.uint8))
                  for ref in refs if ref['kind'] == 'frame']
        return refs, [{'type': 'text', 'text': 'fixture sampled frames'}], frames

    def metadata(self):
        return {'finish_reason': 'stop', 'response_format': 'json_schema',
                'structured_output_degraded': None, 'reasoning_effort': 'low',
                'response_schema_sha256': 'a' * 64}

    def response(self, event=None):
        return {'events': [wire_event() if event is None else event]}, self.metadata()

    def understand(self, directory, retries=1, deadline=None):
        return semantic._understand_context(
            {'id': 'fixture', 'phase': 'review', 'start': 100., 'end': 105.},
            {'identity': {'sha256': 'b' * 64}, 'duration_sec': 200.},
            Path(directory), self.settings(retries), None,
            {'status': 'unknown', 'windows': [], 'sound_events': []},
            time.monotonic() + 30 if deadline is None else deadline)

    def assert_success_audit(self, result, attempts, events=1):
        self.assertEqual(len(result['events']), events)
        self.assertEqual(result['attempts'], attempts)
        self.assertEqual(len(result['retry_history']), attempts - 1)
        self.assertEqual(result['wire_protocol'], 'frame_anchors_v3')
        self.assertIn('wire_response', result)
        self.assertNotIn('PRIVATE_API_KEY', json.dumps(result))

    def test_bad_fact_index_recovers_to_grounded_nonempty_response_with_feedback(self):
        bad = wire_event(); bad['dimensions']['action_value']['fact_indexes'] = [999]
        responses = iter([self.response(bad), self.response()])
        payloads = []
        def request(endpoint, key, payload, timeout):
            payloads.append(copy.deepcopy(payload))
            return next(responses)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler) as sampler, \
             patch.object(semantic, 'request_json', side_effect=request) as remote, \
             patch.object(semantic.time, 'sleep'):
            result = self.understand(directory)
            saved = [json.loads(path.read_text()) for path in Path(directory).glob('*.json')]
        self.assertEqual(remote.call_count, 2)
        self.assertEqual(sampler.call_count, 1)
        self.assert_success_audit(result, 2)
        self.assertEqual(result['wire_response'], {'events': [wire_event()]})
        self.assertEqual(result['events'][0]['observations'][0]['evidence_ids'], ['f1'])
        self.assertEqual(result['events'][0]['dimensions']['action_value']['value'], 3)
        self.assertGreater(len(payloads[1]['messages']), len(payloads[0]['messages']))
        self.assertEqual(payloads[1]['messages'][:2], payloads[0]['messages'][:2])
        self.assertEqual(payloads[1]['response_format'], payloads[0]['response_format'])
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['attempts'], 2)

    def test_unnamed_dimension_array_retries_to_explicit_names_without_guessing(self):
        bad, good = wire_event(), wire_event()
        bad['dimensions'] = list(bad['dimensions'].values())
        good['dimensions'] = [{'dimension': key, **score}
                              for key, score in good['dimensions'].items()]
        good['dimensions'].reverse()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[self.response(bad), self.response(good)]) as remote, \
             patch.object(semantic.time, 'sleep'):
            result = self.understand(directory)
        self.assertEqual(remote.call_count, 2)
        self.assert_success_audit(result, 2)
        self.assertEqual(result['dimension_encodings'], ['named_entries'])
        self.assertEqual(result['wire_response']['events'][0]['dimensions'], good['dimensions'])
        self.assertEqual(result['events'][0]['dimensions']['action_value']['value'], 3)
        self.assertEqual(result['retry_history'][0]['error'], 'EventContractError')

    def test_wrong_top_level_shape_is_retried_not_coerced(self):
        malformed = {'events': [], 'coarse_summary': 'must not be silently discarded'}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[(malformed, self.metadata()), self.response()]) as remote, \
             patch.object(semantic.time, 'sleep'):
            result = self.understand(directory)
        self.assertEqual(remote.call_count, 2)
        self.assert_success_audit(result, 2)

    def test_oversized_integer_is_a_contract_failure_and_can_recover(self):
        bad = wire_event(); bad['confidence'] = 10**400
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[self.response(bad), self.response()]) as remote, \
             patch.object(semantic.time, 'sleep'):
            result = self.understand(directory)
        self.assertEqual(remote.call_count, 2)
        self.assert_success_audit(result, 2)
        self.assertEqual(result['retry_history'][0]['validation_error'], '无效事件置信度')

    def test_contract_diagnostics_keep_safe_gateway_fields_only(self):
        metadata = {**self.metadata(), 'gateway_json_repaired': True,
                    'response_content_encoding': 'fenced_json', 'private_body': 'PRIVATE'}
        error = semantic.EventContractError('无效事件置信度', metadata)
        diagnostics = semantic.failure_diagnostics(error)
        self.assertIs(diagnostics['request']['gateway_json_repaired'], True)
        self.assertEqual(diagnostics['request']['response_content_encoding'], 'fenced_json')
        self.assertNotIn('PRIVATE', json.dumps(diagnostics))

    def test_old_canonical_response_is_not_accepted_as_new_wire_protocol(self):
        canonical = decode_event_response({'events': [wire_event()]}, evidence(), 100., 105.)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', return_value=(canonical, self.metadata())) as remote, \
             patch.object(semantic.time, 'sleep'):
            with self.assertRaises(semantic.EventContractError) as caught:
                self.understand(directory, retries=0)
            self.assertFalse(list(Path(directory).glob('*.json')))
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(caught.exception.attempts, 1)

    def test_disconnect_retries_to_valid_nonempty_or_explicit_empty(self):
        for data in ({'events': [wire_event()]}, {'events': []}):
            with self.subTest(events=len(data['events'])), tempfile.TemporaryDirectory() as directory, \
                 patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
                 patch.object(semantic, 'request_json', side_effect=[
                     RemoteDisconnected('PRIVATE_REMOTE_BODY'), (data, self.metadata())]) as remote, \
                 patch.object(semantic.time, 'sleep'):
                result = self.understand(directory)
            self.assertEqual(remote.call_count, 2)
            self.assert_success_audit(result, 2, len(data['events']))
            self.assertNotIn('PRIVATE_REMOTE_BODY', json.dumps(result['retry_history']))

    def test_connection_reset_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[
                 ConnectionResetError('PRIVATE_SOCKET_DETAIL'), self.response()]) as remote, \
             patch.object(semantic.time, 'sleep'):
            result = self.understand(directory)
        self.assertEqual(remote.call_count, 2)
        self.assert_success_audit(result, 2)

    def test_invalid_json_retries_without_downgrading_response_contract(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[
                 semantic.ResponseContractError('invalid_json', 'stop'), self.response()]) as remote, \
             patch.object(semantic.time, 'sleep'):
            result = self.understand(directory)
        self.assertEqual(remote.call_count, 2)
        self.assert_success_audit(result, 2)
        self.assertIn('invalid_json', json.dumps(result['retry_history']))
        self.assertEqual(remote.call_args.args[2]['response_format']['type'], 'json_schema')
        self.assertTrue(remote.call_args.args[2]['response_format']['json_schema']['strict'])

    def test_transient_http_statuses_retry_but_permanent_errors_do_not(self):
        for status in (408, 429, 500, 502, 503, 400, 401):
            error = HTTPError('https://example.invalid/PRIVATE_URL', status,
                              'PRIVATE_HTTP_DETAIL', {}, io.BytesIO(b'PRIVATE_HTTP_BODY'))
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, \
                 patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
                 patch.object(semantic, 'request_json', side_effect=[error, self.response()]) as remote, \
                 patch.object(semantic.time, 'sleep'):
                if status in (400, 401):
                    with self.assertRaises(HTTPError) as caught: self.understand(directory, retries=3)
                    self.assertEqual(remote.call_count, 1)
                    self.assertEqual(caught.exception.attempts, 1)
                    self.assertFalse(list(Path(directory).glob('*.json')))
                else:
                    result = self.understand(directory)
                    self.assertEqual(remote.call_count, 2)
                    self.assert_success_audit(result, 2)
                    self.assertIn(str(status), json.dumps(result['retry_history']))
                    self.assertNotIn('PRIVATE_', json.dumps(result['retry_history']))

    def test_persistent_invalid_events_exhaust_shared_retry_count_without_caching(self):
        bad = wire_event(); bad['peak_frame'] = 'invented'
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', return_value=self.response(bad)) as remote, \
             patch.object(semantic.time, 'sleep'):
            with self.assertRaises(semantic.EventContractError) as caught:
                self.understand(directory, retries=2)
            self.assertFalse(list(Path(directory).glob('*.json')))
        self.assertEqual(remote.call_count, 3)
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(len(caught.exception.retry_history), 3)

    def test_different_failure_classes_share_one_retry_budget(self):
        bad = wire_event(); bad['dimensions']['action_value']['fact_indexes'] = [-1]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[
                 RemoteDisconnected('PRIVATE'), self.response(bad), self.response()]) as remote, \
             patch.object(semantic.time, 'sleep'):
            with self.assertRaises(semantic.EventContractError) as caught:
                self.understand(directory, retries=1)
            self.assertFalse(list(Path(directory).glob('*.json')))
        self.assertEqual(remote.call_count, 2)
        self.assertEqual(caught.exception.attempts, 2)
        self.assertEqual(len(caught.exception.retry_history), 2)

    def test_persistent_invalid_json_is_a_failure_not_empty_success(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[
                 semantic.ResponseContractError('invalid_json', 'stop'),
                 semantic.ResponseContractError('invalid_json', 'stop')]) as remote, \
             patch.object(semantic.time, 'sleep'):
            with self.assertRaises(semantic.ResponseContractError) as caught:
                self.understand(directory)
            self.assertFalse(list(Path(directory).glob('*.json')))
        self.assertEqual(remote.call_count, 2)
        self.assertEqual(caught.exception.reason, 'invalid_json')
        self.assertEqual(caught.exception.attempts, 2)

    def test_initially_exhausted_deadline_sends_no_request(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json') as remote, \
             patch.object(semantic.time, 'monotonic', return_value=20.), \
             patch.object(semantic.time, 'sleep'):
            with self.assertRaises(TimeoutError): self.understand(directory, deadline=10.)
            self.assertFalse(list(Path(directory).glob('*.json')))
        remote.assert_not_called()

    def test_deadline_exhausted_after_failure_never_sends_second_request(self):
        clock = [0.]
        def failed_request(*args):
            clock[0] = 20.
            raise RemoteDisconnected('PRIVATE_DEADLINE_RESPONSE')
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=failed_request) as remote, \
             patch.object(semantic.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(semantic.time, 'sleep'):
            with self.assertRaises((TimeoutError, RemoteDisconnected)) as caught:
                self.understand(directory, retries=3, deadline=10.)
            self.assertFalse(list(Path(directory).glob('*.json')))
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(caught.exception.retry_history), 1)

    def test_request_timeouts_remain_bounded_by_original_absolute_deadline(self):
        clock = [0.]
        timeouts = []
        def request(endpoint, key, payload, timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                clock[0] = 7.
                raise RemoteDisconnected('transient')
            return self.response()
        def sleep(seconds): clock[0] += seconds
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=request), \
             patch.object(semantic.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(semantic.time, 'sleep', side_effect=sleep):
            result = self.understand(directory, deadline=10.)
        self.assert_success_audit(result, 2)
        self.assertEqual(timeouts[0], 10.)
        self.assertGreater(timeouts[1], 0.)
        self.assertLessEqual(timeouts[1], 3.)

    def test_bounded_map_persists_safe_failure_audit_without_remote_secrets(self):
        bad = wire_event(); bad['dimensions']['action_value']['fact_indexes'] = [999]
        metadata = {**self.metadata(), 'usage': {'private': 'PRIVATE_USAGE'}, 'provider_body': 'PRIVATE_BODY'}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', side_effect=[
                 RemoteDisconnected('PRIVATE_SOCKET'), ({'events': [bad]}, metadata)]), \
             patch.object(semantic.time, 'sleep'):
            results, failures = semantic.bounded_map([{'id': 'fixture'}],
                lambda job: self.understand(directory), 1, time.monotonic() + 30)
            self.assertFalse(list(Path(directory).glob('*.json')))
        self.assertFalse(results)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['attempts'], 2)
        self.assertEqual(len(failures[0]['retry_history']), 2)
        self.assertEqual(failures[0]['status'], 'unreviewed')
        self.assertEqual(failures[0]['response_contract'], 'invalid_event_contract')
        self.assertNotIn('PRIVATE_', json.dumps(failures))


if __name__ == '__main__': unittest.main()
