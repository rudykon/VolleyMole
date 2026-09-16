import contextlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from volleymole import semantic
from volleymole.event_pipeline import settings_from
from volleymole.run_match import argument_parser, validate_event_arguments


class SemanticRequestLimitTests(unittest.TestCase):
    def args(self, *extra):
        parser = argument_parser()
        args = parser.parse_args(['--video', 'unused.mp4', '--no-sound-model', *extra])
        validate_event_arguments(args, parser)
        return args

    def settings(self, **overrides):
        return {'model':'fixture-model', 'endpoint':'https://example.invalid', 'key':'fixture',
                'retries':0, 'timeout':10, 'coarse_fps':2, 'review_fps':8, 'modality':'frames',
                **overrides}

    def sampler(self, source, start, end, fps, width, *args, **kwargs):
        return ([{'id':'frame_0', 'kind':'frame', 'start_sec':start, 'end_sec':start}],
                [{'type':'text', 'text':'fixture frame payload'}],
                [(start, np.zeros((2, 2), dtype=np.uint8)), (end-.04, np.zeros((2, 2), dtype=np.uint8))])

    def understand(self, directory, settings, phase='review'):
        return semantic._understand_context({'id':'fixture', 'phase':phase, 'start':1., 'end':4.},
            {'identity':{'sha256':'a'*64}, 'duration_sec':9.}, directory, settings, None,
            {'status':'unknown', 'windows':[], 'sound_events':[]}, time.monotonic()+30)

    def test_cli_defaults_explicit_settings_and_legacy_namespace(self):
        args = self.args('--semantic-max-tokens', '2048', '--semantic-frame-width', '224', '--semantic-reasoning-effort', 'low')
        configured = settings_from(args)
        self.assertEqual(configured['max_tokens'], 2048)
        self.assertEqual(configured['frame_width'], 224)
        self.assertEqual(configured['reasoning_effort'], 'low')
        defaults = self.args()
        self.assertEqual(defaults.semantic_max_tokens, 4096)
        self.assertIsNone(defaults.semantic_frame_width)
        self.assertIsNone(defaults.semantic_reasoning_effort)
        del defaults.semantic_max_tokens, defaults.semantic_frame_width
        configured = settings_from(defaults)
        self.assertEqual(configured['max_tokens'], 4096)
        self.assertIsNone(configured['frame_width'])

    def test_invalid_cli_limits_rejected(self):
        for option, value in (('--semantic-max-tokens','255'), ('--semantic-max-tokens','16385'),
                              ('--semantic-frame-width','63'), ('--semantic-frame-width','4097')):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(option, value)

    def test_limits_reach_real_request_payload_without_changing_fps_or_context(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler) as sampler, \
             patch.object(semantic, 'request_json', return_value=({'events':[]}, {'usage':{}, 'finish_reason':'stop'})) as request:
            result = self.understand(Path(directory), self.settings(max_tokens=2048, frame_width=224, reasoning_effort='low'))
        self.assertEqual(request.call_args.args[2]['max_tokens'], 2048)
        self.assertEqual(request.call_args.args[2]['reasoning_effort'], 'low')
        self.assertEqual(request.call_args.args[2]['response_format']['type'], 'json_schema')
        self.assertTrue(request.call_args.args[2]['response_format']['json_schema']['strict'])
        self.assertEqual(sampler.call_args.args[1:5], (1., 4., 8, 224))
        self.assertEqual(result['requested_fps'], 8)
        self.assertEqual(result['sample_times_sec'], [1., 3.96])

    def test_optional_cap_never_upscales_default_width_and_legacy_settings_work(self):
        for phase, settings, expected in (
                ('coarse', self.settings(), 512), ('review', self.settings(), 768),
                ('coarse', self.settings(frame_width=1024), 512),
                ('review', self.settings(frame_width=1024), 768),
                ('coarse', self.settings(frame_width=224), 224)):
            with self.subTest(phase=phase, expected=expected), tempfile.TemporaryDirectory() as directory, \
                 patch.object(semantic, 'sampled_evidence', side_effect=self.sampler) as sampler, \
                 patch.object(semantic, 'request_json', return_value=({'events':[]}, {'usage':{}, 'finish_reason':'stop'})) as request:
                self.understand(Path(directory), settings, phase)
                self.assertEqual(sampler.call_args.args[4], expected)
                self.assertEqual(request.call_args.args[2]['max_tokens'], 4096)

    def test_token_or_width_change_invalidates_cache_and_identical_config_reuses(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', return_value=({'events':[]}, {'usage':{}, 'finish_reason':'stop'})) as request:
            first = self.understand(Path(directory), self.settings(max_tokens=2048, frame_width=224))
            second = self.understand(Path(directory), self.settings(max_tokens=1024, frame_width=224))
            third = self.understand(Path(directory), self.settings(max_tokens=1024, frame_width=192))
            reused = self.understand(Path(directory), self.settings(max_tokens=1024, frame_width=192))
        self.assertEqual(len({r['signature'] for r in (first, second, third)}), 3)
        self.assertEqual(request.call_count, 3)
        self.assertTrue(reused['cached'])
        self.assertEqual(reused['signature'], third['signature'])

    def test_reasoning_effort_and_schema_implementation_invalidate_cache(self):
        original_digest = semantic.digest
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', return_value=({'events': []}, {'finish_reason': 'stop'})) as request:
            first = self.understand(Path(directory), self.settings())
            second = self.understand(Path(directory), self.settings(reasoning_effort='low'))
            with patch.object(semantic, 'digest', side_effect=lambda path:
                    'changed-schema' if path.name == 'event_schema.py' else original_digest(path)):
                third = self.understand(Path(directory), self.settings(reasoning_effort='low'))
                reused = self.understand(Path(directory), self.settings(reasoning_effort='low'))
        self.assertEqual(request.call_count, 3)
        self.assertEqual(len({r['signature'] for r in (first, second, third)}), 3)
        self.assertTrue(reused['cached'])

    def test_degraded_invalid_response_stays_failed_with_safe_diagnostics(self):
        data = {'events': [], 'coarse_summary': 'PRIVATE_RESPONSE'}
        metadata = {'finish_reason': 'stop', 'response_format': 'json_schema',
                    'structured_output_degraded': 'model_unsupported', 'usage': {'untrusted': 'PRIVATE_RESPONSE'}}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(semantic, 'sampled_evidence', side_effect=self.sampler), \
             patch.object(semantic, 'request_json', return_value=(data, metadata)) as request:
            results, failures = semantic.bounded_map([{'id': 'fixture'}],
                lambda job: self.understand(Path(directory), self.settings()), 1, time.monotonic()+10)
            self.assertFalse(list(Path(directory).glob('*.json')))
        self.assertEqual(request.call_count, 1)
        self.assertFalse(results)
        self.assertEqual(failures[0]['response_contract'], 'invalid_event_contract')
        self.assertEqual(failures[0]['request']['structured_output_degraded'], 'model_unsupported')
        self.assertNotIn('PRIVATE_RESPONSE', str(failures))


if __name__ == '__main__': unittest.main()
