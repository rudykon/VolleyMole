"""Request-cache/protocol tests; simulated responses are not quality evidence."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from volleymole.common import read_json, save_json
from volleymole.llm_transport import ResponseContractError, TransportError
from volleymole.replay_evidence import _encode
from volleymole.replay_requests import request_phase, request_implementation
from volleymole.replay_review_schema import decode_scan


def sample(source, start, end, fps, width, **kwargs):
    # Static images intentionally need no native high-resolution detail decode.
    url = _encode(np.zeros((32,64,3), dtype=np.uint8))
    evidence = []; content = []; index = 0
    while start+index/fps < end:
        when = start+index/fps
        fid = f'frame_{index:05d}'
        evidence.append(dict(id=fid,kind='frame',start_sec=when,end_sec=when))
        content.extend([dict(type='text',text=f'{fid} source_sec={when:.6f}'),
                        dict(type='image_url',image_url=dict(url=url))])
        index += 1
    return evidence, content, []


class ReplayRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)/'cache'
        self.source = dict(path='unused static fake',identity={'sha256':'source-A'},
                           nominal_fps='30/1',rotation=0,start_sec=0.,duration_sec=60.)
        self.settings = dict(endpoint='https://example.invalid/v1',model='vision',key='NEVER_CACHE_THIS_KEY',
            timeout=20.,retries=0,max_tokens=2048,reasoning_effort='low',scan_fps=4.,review_fps=8.,
            width=768,max_expansions=2,max_candidates=3,min_confidence=.65,max_context_sec=18.)
        self.job = dict(phase='scan',start=10.,end=12.,index=0)
        self.valid = dict(candidates=[dict(action='spike',preparation_frame_id='frame_00000',
            peak_frame_id='frame_00002',result_frame_id='frame_00005',excitement=4,confidence=.8,
            observation='Visible preparation, contact and subsequent ball path.')],uncertainty='')
        self.provider = Mock(side_effect=lambda *args: (copy.deepcopy(self.valid),dict(finish_reason='stop')))
        self.sampler = Mock(side_effect=sample)

    def run_request(self, *, settings=None, job=None, source=None, decoder=decode_scan, deadline=None):
        return request_phase(source or self.source, job or self.job, settings or self.settings,
            self.cache, deadline or time.monotonic()+30, decoder=decoder,
            request_fn=self.provider, sample_fn=self.sampler)

    def test_identical_visual_request_reuses_across_policy_budget_changes(self):
        first, one = self.run_request()
        changed = dict(self.settings,max_expansions=3,max_candidates=6,min_confidence=.9,
                       max_context_sec=25.,timeout=5.,retries=2,key='ROTATED_VALID_KEY')
        second, two = self.run_request(settings=changed)
        self.assertEqual(first, second); self.assertTrue(two['cached'])
        self.assertEqual(one['signature'], two['signature'])
        self.assertEqual(self.provider.call_count,1); self.assertEqual(self.sampler.call_count,1)
        serialized = Path(two['artifact']).read_text()
        self.assertNotIn('NEVER_CACHE_THIS_KEY',serialized)
        self.assertNotIn('ROTATED_VALID_KEY',serialized)
        self.assertNotIn('max_expansions',serialized)
        self.assertNotIn('replay_review.py',two['identity']['implementation'])
        self.assertIn('replay_requests.py',two['identity']['implementation'])
        self.assertEqual(two['visual_context']['frame_count'],8)
        self.assertEqual(len(two['image_sha256']),9)

    def test_model_endpoint_image_and_generation_parameters_invalidate_request(self):
        _, original = self.run_request()
        changed_values = [('model','vision-other'),('endpoint','https://another.invalid/v1'),
                          ('width',512),('scan_fps',8.),('max_tokens',4096),
                          ('reasoning_effort','high'),('overview_frames',6),('detail_limit',0)]
        for key, value in changed_values:
            with self.subTest(key=key):
                _, audit = self.run_request(settings={**self.settings,key:value})
                self.assertFalse(audit['cached'])
                self.assertNotEqual(audit['signature'],original['signature'])
        self.assertEqual(self.provider.call_count,1+len(changed_values))

    def test_inactive_phase_fps_does_not_invalidate_scan(self):
        self.run_request()
        _, audit = self.run_request(settings={**self.settings,'review_fps':16.,'verify_fps':12.})
        self.assertTrue(audit['cached'])

    def test_complete_job_including_hints_and_source_clock_is_bound(self):
        _, original = self.run_request()
        jobs = [dict(self.job,end=12.5),dict(self.job,index=1),
                dict(self.job,boundary_problem='Need to see the next actual touch.'),
                dict(self.job,spatial_hints=[])]
        for job in jobs:
            _, audit = self.run_request(job=job)
            self.assertNotEqual(audit['signature'],original['signature'])
        for source in (dict(self.source,start_sec=.1),dict(self.source,rotation=90),
                       dict(self.source,identity={'sha256':'source-B'})):
            _, audit = self.run_request(source=source)
            self.assertNotEqual(audit['signature'],original['signature'])

    def test_only_current_phase_prompt_is_cache_bound(self):
        implementation = request_implementation('scan')
        self.assertIn('prompts/replay_scan.md',implementation)
        self.assertNotIn('prompts/replay_review.md',implementation)
        self.assertNotIn('replay.py',implementation)
        self.assertNotIn('replay_review.py',implementation)
        self.assertNotIn('replay_stage.py',implementation)
        with self.assertRaisesRegex(ValueError,'replay_request_phase'):
            request_implementation('../anything')

    def test_protocol_failure_retries_without_repairing_raw_response(self):
        bad = copy.deepcopy(self.valid); bad['candidates'][0]['peak_frame_id'] = 2
        preserved = copy.deepcopy(bad)
        self.provider.side_effect = [(bad,{}),(copy.deepcopy(self.valid),{})]
        _, audit = self.run_request(settings={**self.settings,'retries':1})
        self.assertEqual(self.provider.call_count,2)
        self.assertEqual(bad,preserved)
        self.assertEqual(audit['raw'],self.valid)
        self.assertEqual(len(audit['retry_history']),1)
        self.assertEqual(audit['retry_history'][0]['validation_error'],'replay_unknown_frame')
        # Only a correction instruction follows the original pictures; failed
        # provider output is neither laundered into the request nor persisted.
        wire = self.provider.call_args.args[2]
        self.assertEqual(len(wire['messages']),3)
        self.assertNotIn('peak_frame_id\": 2',json.dumps(wire['messages'],ensure_ascii=False))

    def test_failed_contract_never_becomes_empty_success_or_saved_cache(self):
        self.provider.side_effect = ResponseContractError('invalid_json')
        with self.assertRaises(ResponseContractError):
            self.run_request(settings={**self.settings,'retries':1})
        self.assertEqual(self.provider.call_count,2)
        self.assertFalse(list(self.cache.glob('requests/*.json')))

    def test_retryable_transport_is_bounded_and_diagnostics_redacted(self):
        self.provider.side_effect = [TimeoutError('secret provider body SHOULD_NEVER_APPEAR'),(self.valid,{})]
        _, audit = self.run_request(settings={**self.settings,'retries':1})
        self.assertEqual(self.provider.call_count,2)
        self.assertNotIn('SHOULD_NEVER_APPEAR',json.dumps(audit))
        self.assertLessEqual(self.provider.call_args.args[3],self.settings['timeout'])
        self.provider.side_effect = TransportError('authentication_failed',retryable=False,http_status=401)
        with self.assertRaises(TransportError):
            self.run_request(job=dict(self.job,index=2),settings={**self.settings,'retries':3})
        self.assertEqual(self.provider.call_count,3)

    def test_shared_deadline_stops_before_sampling_or_provider_call(self):
        with self.assertRaisesRegex(TimeoutError,'replay_review_deadline'):
            self.run_request(deadline=time.monotonic()-1)
        self.sampler.assert_not_called(); self.provider.assert_not_called()

    def test_sampling_gap_or_out_of_scope_frame_never_reaches_provider(self):
        def broken(*args,**kwargs):
            evidence,content,frames = sample(*args,**kwargs)
            evidence[-1]['start_sec'] = args[2]+.1
            return evidence,content,frames
        self.sampler.side_effect = broken
        with self.assertRaisesRegex(ValueError,'replay_sampling_gap'):
            self.run_request()
        self.provider.assert_not_called()

    def test_cache_raw_response_is_revalidated_on_every_use(self):
        _, audit = self.run_request()
        saved = read_json(audit['artifact']); saved['raw']['candidates'][0]['confidence'] = '0.8'
        save_json(audit['artifact'],saved)
        _, renewed = self.run_request()
        self.assertFalse(renewed['cached']); self.assertEqual(self.provider.call_count,2)
        self.assertEqual(renewed['raw'],self.valid)

    def test_candidate_old_frame_ids_are_not_sent_as_new_sequence_anchors(self):
        candidate = dict(peak_sec=10.5,preparation_frame_id='frame_OLD_01',peak_frame_id='frame_OLD_02',action='spike')
        self.run_request(job=dict(self.job,candidate=candidate,proposal={'target_start_sec':10.2}))
        metadata = json.loads(self.provider.call_args.args[2]['messages'][1]['content'][0]['text'])
        self.assertEqual(metadata['candidate'],dict(peak_sec=10.5,action='spike'))
        self.assertEqual(metadata['proposal'],{'target_start_sec':10.2})
        self.assertFalse(metadata['audio_available'])


if __name__ == '__main__':
    unittest.main()
