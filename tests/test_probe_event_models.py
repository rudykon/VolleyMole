import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import probe_event_models as probe


class ModelProbeTests(unittest.TestCase):
    def test_same_frames_compare_models_and_keep_empty_and_failed_results(self):
        def sample(source, start, end, fps, width, **kwargs):
            evidence = [{'id': f'frame_{i}', 'kind': 'frame',
                'start_sec': start+i/fps, 'end_sec': start+i/fps} for i in range(int((end-start)*fps))]
            return evidence, [{'type': 'text', 'text': 'fixture'}], [(e['start_sec'], None) for e in evidence]
        def request(endpoint, key, payload, timeout):
            self.assertEqual(payload['reasoning_effort'], 'low')
            self.assertEqual(payload['response_format']['type'], 'json_schema')
            self.assertTrue(payload['response_format']['json_schema']['strict'])
            if payload['model'] == 'limited':
                raise HTTPError(endpoint, 400, 'Bad Request', {}, io.BytesIO(
                    b'This model\'s maximum context length is 8192 tokens and your request has 9999 input tokens. PRIVATE_ECHO'))
            return {'events': []}, {'finish_reason': 'stop', 'usage': {'prompt_tokens': 100}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ['--run', str(root/'run'), '--output', str(root/'output'),
                '--models', 'valid', 'limited', '--start', '1', '--end', '4', '--reasoning-effort', 'low']
            with patch.object(probe, 'read_json', side_effect=lambda path:
                    {'source': {'duration_sec': 10, 'identity': {'sha256': 'fixture'}}}
                    if path.name == 'match_manifest.json' else
                    {'llm': {'base_url': 'https://example.invalid/v1', 'api_key': 'test-secret'}}), \
                 patch.object(probe, 'urlopen', return_value=io.BytesIO(json.dumps(
                    {'data': [{'id': 'valid'}, {'id': 'limited'}]}).encode())), \
                 patch.object(probe, 'sampled_evidence', side_effect=sample), \
                 patch.object(probe, 'request_json', side_effect=request), \
                 contextlib.redirect_stdout(io.StringIO()):
                probe.main(args)
            text = (root/'output/report.json').read_text()
            report = json.loads(text)
            self.assertFalse(report['audio_uploaded'])
            self.assertEqual([r['frame_count'] for r in report['results']], [6, 6, 24, 24])
            for result in report['results']:
                self.assertEqual(result['reasoning_effort'], 'low')
                if result['model'] == 'valid':
                    self.assertEqual((result['status'], result['event_count']), ('valid', 0))
                else:
                    self.assertEqual(result['status'], 'failed')
                    self.assertEqual(result['context_limit_tokens'], 8192)
                    self.assertEqual(result['input_tokens_reported'], 9999)
            self.assertNotIn('PRIVATE_ECHO', text)
            self.assertNotIn('test-secret', text)

    def test_invalid_limits_rejected_before_network_access(self):
        for option, value in (('--width', '0'), ('--max-tokens', '255'),
                              ('--timeout', 'nan'), ('--timeout', '0')):
            with self.subTest(option=option, value=value), tempfile.TemporaryDirectory() as directory, \
                 patch.object(probe, 'urlopen') as network, contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                try:
                    probe.main(['--run', 'unused', '--models', 'unused', '--start', '0', '--end', '1',
                        '--output', str(Path(directory)/'output'), option, value])
                finally:
                    network.assert_not_called()

    def test_partial_measurement_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            output = Path(directory)
            artifact = output/'probe_00.json'
            artifact.write_text('original measurement')
            with self.assertRaises(SystemExit):
                probe.main(['--run', 'unused', '--models', 'unused', '--start', '0', '--end', '1',
                    '--output', str(output)])
            self.assertEqual(artifact.read_text(), 'original measurement')


if __name__ == '__main__':
    unittest.main()
