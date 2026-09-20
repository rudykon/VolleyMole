"""Failed model JSON is auditable, bounded, redacted, and never a success cache."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import test_replay_requests as fixtures

from volleymole.common import read_json
from volleymole.replay_requests import fingerprint, request_phase
from volleymole.replay_review_schema import decode_scan, decode_review


class ReplayFailedRequestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)/'cache'
        self.source = dict(path='unused',identity={'sha256':'source-A'},
                           nominal_fps='30/1',rotation=0,start_sec=0.,duration_sec=60.)
        self.settings = dict(endpoint='https://example.invalid/v1',model='vision',key='SECRET_DIAGNOSTIC_KEY',
            timeout=20.,retries=1,max_tokens=2048,reasoning_effort='low',scan_fps=4.,review_fps=8.,width=768)
        self.job = dict(phase='scan',start=10.,end=12.,index=0)
        self.valid = dict(candidates=[dict(action='spike',preparation_frame_id='frame_00000',
            peak_frame_id='frame_00002',result_frame_id='frame_00005',excitement=4,confidence=.8,
            observation='Visible preparation and contact.')],uncertainty='')
        self.bad = copy.deepcopy(self.valid); del self.bad['candidates'][0]['confidence']
        self.provider = Mock()

    def run_request(self, *, decoder=decode_scan, settings=None, job=None):
        return request_phase(self.source,job or self.job,settings or self.settings,self.cache,
            time.monotonic()+30,decoder=decoder,request_fn=self.provider,sample_fn=fixtures.sample)

    def failure_records(self):
        return [read_json(path) for path in sorted(self.cache.glob('failed_requests/**/*.json'))]

    def test_schema_failures_preserve_model_json_but_not_wire_or_credentials(self):
        metadata = dict(finish_reason='stop',usage={'prompt_tokens':10,'completion_tokens':2,'secret':'HIDDEN_USAGE'},
                        authorization='SECRET_DIAGNOSTIC_KEY',body='ARBITRARY_PROVIDER_BODY')
        self.provider.side_effect = [(copy.deepcopy(self.bad),metadata),(copy.deepcopy(self.bad),metadata)]
        with self.assertRaisesRegex(ValueError,'replay_response_fields'):
            self.run_request()
        records = self.failure_records()
        self.assertEqual(len(records),2)
        self.assertEqual({r['attempt'] for r in records},{1,2})
        for record in records:
            self.assertEqual(record['status'],'failed'); self.assertIs(record['reusable'],False)
            self.assertEqual(record['raw'],self.bad)
            self.assertEqual(record['validation_error'],'replay_response_fields')
            self.assertEqual(record['source_identity_sha256'],fingerprint(self.source['identity']))
            self.assertEqual(record['job_sha256'],fingerprint(self.job))
            self.assertEqual(record['request']['usage'],{'prompt_tokens':10,'completion_tokens':2})
        serialized = json.dumps(records)
        for private in ('SECRET_DIAGNOSTIC_KEY','HIDDEN_USAGE','ARBITRARY_PROVIDER_BODY','data:image','base64,'):
            self.assertNotIn(private,serialized)
        self.assertFalse(list(self.cache.glob('requests/*.json')))
        self.assertEqual(self.provider.call_count,2)

    def test_retry_names_only_fixed_local_validation_code(self):
        self.provider.side_effect = [(copy.deepcopy(self.bad),{}),(copy.deepcopy(self.valid),{})]
        _,audit = self.run_request()
        retry = self.provider.call_args.args[2]['messages'][2]['content']
        self.assertIn('replay_response_fields',retry)
        self.assertEqual(len(self.failure_records()),1)
        self.assertNotIn(json.dumps(self.bad),retry)
        self.assertEqual(audit['raw'],self.valid)

    def test_unrecognized_exception_text_is_not_reflected_into_retry_or_disk(self):
        self.provider.side_effect = [(copy.deepcopy(self.bad),{}),(copy.deepcopy(self.valid),{})]
        first = True
        def decoder(raw,evidence):
            nonlocal first
            if first:
                first=False
                raise ValueError('replay_PRIVATE_EXCEPTION_TEXT SECRET_DIAGNOSTIC_KEY')
            return decode_scan(raw,evidence)
        _,audit = self.run_request(decoder=decoder)
        records = self.failure_records(); self.assertEqual(len(records),1)
        self.assertEqual(records[0]['validation_error'],'replay_invalid_contract')
        serialized = json.dumps([records,audit,self.provider.call_args.args[2]])
        self.assertNotIn('PRIVATE_EXCEPTION_TEXT',serialized)
        self.assertNotIn('SECRET_DIAGNOSTIC_KEY',serialized)

    def test_transport_failure_cannot_reuse_previous_attempt_raw_or_metadata(self):
        self.provider.side_effect = [(copy.deepcopy(self.bad),{'finish_reason':'stop'}),
                                    TimeoutError('DO_NOT_SAVE_PROVIDER_BODY')]
        with self.assertRaises(TimeoutError):
            self.run_request()
        records = sorted(self.failure_records(),key=lambda r:r['attempt'])
        self.assertEqual(len(records),2)
        self.assertEqual(records[0]['raw'],self.bad)
        self.assertIsNone(records[1]['raw'])
        self.assertEqual(records[1]['request'],{})
        self.assertNotIn('DO_NOT_SAVE_PROVIDER_BODY',json.dumps(records))

    def test_failed_artifact_is_not_reused_as_a_success_on_resume(self):
        self.provider.side_effect = [(copy.deepcopy(self.bad),{}),(copy.deepcopy(self.valid),{})]
        with self.assertRaises(ValueError):
            self.run_request(settings={**self.settings,'retries':0})
        _,audit = self.run_request(settings={**self.settings,'retries':0})
        self.assertFalse(audit['cached'])
        self.assertEqual(self.provider.call_count,2)
        self.assertEqual(len(self.failure_records()),1)
        self.assertEqual(len(list(self.cache.glob('requests/*.json'))),1)

    def test_model_echoed_sensitive_fields_are_redacted_in_failed_json(self):
        raw = dict(self.bad,api_key='SECRET_DIAGNOSTIC_KEY',wire={'messages':[{'type':'image_url',
            'image_url':{'url':'data:image/jpeg;base64,'+'A'*400}}]},
            note='SECRET_DIAGNOSTIC_KEY returned in invalid JSON')
        self.provider.side_effect = [(raw,{})]
        with self.assertRaises(ValueError):
            self.run_request(settings={**self.settings,'retries':0})
        records = self.failure_records(); self.assertEqual(len(records),1)
        serialized = json.dumps(records)
        self.assertNotIn('SECRET_DIAGNOSTIC_KEY',serialized)
        self.assertNotIn('data:image',serialized)
        self.assertNotIn('A'*400,serialized)
        self.assertTrue(records[0]['raw_redacted'])
        # Redaction applies only to diagnostics, never to the caller's response.
        self.assertEqual(raw['api_key'],'SECRET_DIAGNOSTIC_KEY')

    def test_semantic_rejection_is_not_a_protocol_failure_or_retried_for_yes(self):
        raw = dict(verdict='reject',action='set',need_before=False,need_after=False,
            start_complete=False,end_complete=False,excitement=1,confidence=.9,
            preparation_observation='',contact_observation='',result_observation='',
            reason='Only a normal set is visible.',uncertainty='')
        for name in ('lead','preparation','peak','result','tail'):raw[name+'_frame_id']=None
        self.provider.side_effect = [(raw,{})]
        decoded,_ = self.run_request(decoder=decode_review,job=dict(phase='review',start=10.,end=12.))
        self.assertEqual(decoded['verdict'],'reject')
        self.assertEqual(self.provider.call_count,1)
        self.assertEqual(self.failure_records(),[])


if __name__ == '__main__':
    unittest.main()
