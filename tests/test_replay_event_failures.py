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
import replay_event_failures as replay
from volleymole.llm_transport import TransportError
from volleymole.semantic import EventContractError


class ReplayEventFailureTests(unittest.TestCase):
    def fixtures(self, root):
        run = root/'run'
        run.mkdir()
        source = {'path': str(root/'video.mp4'), 'duration_sec': 800,
                  'width': 1920, 'height': 1080, 'identity': {'sha256': 'fixture'}}
        replay.save_json(run/'event_timeline.json', {'source': source, 'requests': []})
        replay.save_json(run/'match_manifest.json', {'source': source, 'rallies': [
            {'rally_id': 'rally_0013', 'start_sec': 164.5, 'end_sec': 183,
             'safe_start_sec': 162.7, 'safe_end_sec': 185, 'actions': [], 'action_events': []}]})
        replay.save_json(run/'audio_events.json', {'status': 'complete', 'windows': [],
                                                 'sound_events': [], 'action_model': {}})
        (run/'analytics').mkdir()
        (run/'analytics/detections.jsonl').write_text('{"time_s": 0}\n{"time_s": 1}\n')
        (run/'tracking').mkdir()
        (run/'tracking/ball.csv').write_text('Frame,X,Y,Visibility\n0,10,20,1\n1,nan,30,1\n')
        replay.save_json(root/'llm_api.json', {'llm': {'base_url': 'https://example.invalid/v1',
                                                     'api_key': 'PRIVATE_KEY'}})
        return run

    def test_job_recovery_and_recorded_requests_take_precedence(self):
        timeline = {'source': {'duration_sec': 800}, 'requests': [
            {'job': {'id': 'rally_0013', 'start': 160, 'end': 190, 'phase': 'review',
                     'candidate': {'saved': True}}}]}
        manifest = {'rallies': [{'rally_id': 'rally_0030', 'start_sec': 343.5, 'end_sec': 349,
                                'safe_start_sec': 342, 'safe_end_sec': 351,
                                'actions': [], 'action_events': []}]}
        jobs = replay.restore_jobs(timeline, manifest, ['coarse_00013', 'rally_0013', 'rally_0030'])
        self.assertEqual((jobs[0]['start'], jobs[0]['end']), (260, 284))
        self.assertEqual(jobs[1]['candidate'], {'saved': True})
        self.assertEqual((jobs[2]['start'], jobs[2]['end']), (340, 354))
        self.assertEqual(jobs[2]['candidate']['start_sec'], 343.5)
        jobs[1]['candidate']['saved'] = False
        self.assertTrue(timeline['requests'][0]['job']['candidate']['saved'])

    def test_recovery_rejects_unknown_duplicate_oversized_or_outside_jobs(self):
        timeline = {'source': {'duration_sec': 800}, 'requests': [
            {'job': {'id': 'too_long', 'phase': 'review', 'start': 0, 'end': 91}},
            {'job': {'id': 'outside', 'phase': 'coarse', 'start': -1, 'end': 2}}]}
        for names in (['missing'], ['../unsafe'], ['coarse_99999'],
                      ['coarse_00000', 'coarse_00000'], ['too_long'], ['outside']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                replay.restore_jobs(timeline, {'rallies': []}, names)

    def test_records_match_visibility_and_coordinate_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.fixtures(Path(directory))
            records, signature = replay.load_records(run, {'width': 1920, 'height': 1080})
            self.assertEqual(records[0]['event_ball'], [10, 20])
            self.assertNotIn('event_ball', records[1])
            self.assertEqual(len(signature), 2)

    def test_replays_with_new_cache_only_glm_and_safe_diagnostics(self):
        captured = []
        def understand(job, source, cache, settings, audio, features, deadline, records):
            captured.append((job['id'], cache, settings['model'], records))
            self.assertEqual(settings['modality'], 'frames')
            self.assertEqual(settings['key'], 'PRIVATE_KEY')
            self.assertIsNone(audio)
            self.assertGreater(deadline, 0)
            if job['phase'] == 'review':
                error = EventContractError('事实不在事件上下文内', {
                    'finish_reason': 'stop', 'response_format': 'json_schema',
                    'structured_output_degraded': 'PRIVATE_ECHO'})
                error.attempts = 2
                raise error
            return {'events': [], 'attempts': 2, 'cached': False,
                    'wire_protocol': 'frame_anchors_v2',
                    'retry_history': [{'attempt': 1, 'error': 'TransportError',
                        'transport_error': 'connection_error', 'raw_body': 'PRIVATE_ECHO'}],
                    'request': {'finish_reason': 'stop', 'extra': 'PRIVATE_ECHO',
                                'gateway_json_repaired': True,
                                'response_content_encoding': 'fenced_json'}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixtures(root)
            output = root/'new-output'
            stdout = io.StringIO()
            with patch.object(replay, 'ROOT', root), \
                 patch.object(replay, 'understand_context', side_effect=understand), \
                 contextlib.redirect_stdout(stdout):
                report = replay.main(['--run', str(run), '--output', str(output),
                                      '--jobs', 'coarse_00013', 'rally_0013'])
            self.assertEqual(report['status'], 'completed_with_failures')
            self.assertFalse(report['audio_uploaded'])
            self.assertFalse(report['semantic_cache_reused'])
            self.assertEqual(report['results'][0]['event_count'], 0)
            self.assertEqual(report['results'][0]['status'], 'valid')
            self.assertEqual(report['results'][0]['wire_protocol'], 'frame_anchors_v2')
            self.assertEqual(report['results'][0]['retry_history'], [{'attempt': 1,
                'error': 'TransportError', 'transport_error': 'connection_error'}])
            self.assertTrue(report['results'][0]['request']['gateway_json_repaired'])
            self.assertEqual(report['results'][0]['request']['response_content_encoding'], 'fenced_json')
            self.assertEqual(report['results'][1]['validation_error'], '事实不在事件上下文内')
            self.assertEqual(report['results'][1]['attempts'], 2)
            self.assertFalse(report['replay_scope']['byte_identical_payload'])
            self.assertIsNone(report['replay_scope']['same_source_and_context'])
            self.assertEqual(report['results'][0]['context_origin'], 'default_24_4_reconstruction')
            self.assertEqual(report['results'][1]['context_origin'], 'manifest_rally_reconstruction')
            self.assertIn('global sound IDs', report['replay_scope']['coarse_sound_evidence'])
            self.assertIn('llm_transport.py', report['code_sha256'])
            self.assertTrue(all(item[1] == output/'event_cache' and item[2] == replay.MODEL for item in captured))
            self.assertIsNone(next(item[3] for item in captured if item[0].startswith('coarse')))
            serialized = (output/'summary.json').read_text() + stdout.getvalue()
            self.assertNotIn('PRIVATE_KEY', serialized)
            self.assertNotIn('PRIVATE_ECHO', serialized)

    def test_existing_even_empty_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(replay, 'understand_context') as request, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            try:
                replay.main(['--run', 'unused', '--jobs', 'coarse_00000', '--output', directory])
            finally:
                request.assert_not_called()

    def test_invalid_options_rejected_before_io_or_requests(self):
        cases = [('--model', 'kimi'), ('--timeout', 'nan'), ('--timeout', '0'),
                 ('--retries', '-1'), ('--concurrency', '0'), ('--width', '0'),
                 ('--max-review-sec', 'inf'), ('--max-tokens', '255')]
        for option, value in cases:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory, \
                 patch.object(replay, 'read_json') as read, \
                 patch.object(replay, 'understand_context') as request, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                try:
                    replay.main(['--run', 'unused', '--jobs', 'coarse_00000',
                                 '--output', str(Path(directory)/'out'), option, value])
                finally:
                    read.assert_not_called()
                    request.assert_not_called()

    def test_failed_transport_metadata_never_reads_or_copies_body(self):
        error = HTTPError('https://example.invalid', 503, 'PRIVATE_KEY', {}, io.BytesIO(b'PRIVATE_ECHO'))
        with patch.object(error, 'read') as read:
            result = replay.failure_diagnostics(error)
        read.assert_not_called()
        self.assertEqual(result, {'error': 'HTTPError', 'http_status': 503, 'attempts': None})
        malformed = EventContractError('PRIVATE_ECHO', {'finish_reason': 'stop'})
        self.assertEqual(replay.failure_diagnostics(malformed)['validation_error'], 'other_event_contract_error')

    def test_transport_reasons_status_and_retry_history_are_allowlisted(self):
        error = TransportError('http_error', retryable=True, http_status=503)
        error.attempts = 2
        error.retry_history = [
            {'attempt': 1, 'error': 'TransportError', 'transport_error': 'http_error',
             'http_status': 503, 'private': 'PRIVATE_ECHO'},
            {'attempt': 2, 'error': 'PRIVATE_ECHO', 'transport_error': 'PRIVATE_ECHO',
             'http_status': 'PRIVATE_ECHO', 'response_contract': 'PRIVATE_ECHO',
             'validation_error': 'PRIVATE_ECHO', 'request': {'gateway_json_repaired': 'PRIVATE_ECHO',
                 'response_content_encoding': 'PRIVATE_ECHO'}}]
        result = replay.failure_diagnostics(error)
        self.assertEqual(result['transport_error'], 'http_error')
        self.assertEqual(result['http_status'], 503)
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['retry_history'][0]['http_status'], 503)
        self.assertEqual(result['retry_history'][1]['transport_error'], 'other')
        self.assertEqual(result['retry_history'][1]['error'], 'OtherError')
        self.assertNotIn('PRIVATE_ECHO', json.dumps(result))

    def test_new_wire_validation_categories_remain_distinguishable(self):
        for reason in ('锚点事件字段不完整', '帧锚必须引用本上下文的画面证据',
                       '事实辅助证据必须为本上下文的声音或运动测量',
                       '事实帧时间与辅助证据不对应', '评分引用了不存在的事实索引',
                       '非未知评分必须引用事实且取值为 0–4'):
            with self.subTest(reason=reason):
                error = EventContractError(reason, {'finish_reason': 'stop'})
                self.assertEqual(replay.failure_diagnostics(error)['validation_error'], reason)

    def test_retry_history_rejects_untyped_and_unbounded_values(self):
        self.assertEqual(replay.safe_retry_history('PRIVATE_ECHO'), [])
        self.assertEqual(replay.safe_retry_history([None, 'PRIVATE_ECHO']), [])
        result = replay.safe_retry_history([{'attempt': True, 'error': 'TimeoutError'}]*105)
        self.assertEqual(len(result), 100)
        self.assertNotIn('attempt', result[0])

    def test_cli_exit_code_distinguishes_success_and_failed_replays(self):
        for status, expected in (('completed', 0), ('completed_with_failures', 1), ('running', 1)):
            with self.subTest(status=status), \
                 patch.object(replay, 'main', return_value={'status': status}) as main:
                self.assertEqual(replay.cli_main(['--jobs', 'fixture']), expected)
                main.assert_called_once_with(['--jobs', 'fixture'])

    def test_context_identity_is_confirmed_only_for_all_recorded_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixtures(root)
            timeline = replay.read_json(run/'event_timeline.json')
            timeline['requests'] = [{'job': {'id': 'coarse_00013', 'start': 260,
                                             'end': 284, 'phase': 'coarse'}}]
            replay.save_json(run/'event_timeline.json', timeline)
            with patch.object(replay, 'ROOT', root), \
                 patch.object(replay, 'understand_context', return_value={
                     'events': [], 'attempts': 1, 'cached': False}), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = replay.main(['--run', str(run), '--output', str(root/'new-output'),
                                      '--jobs', 'coarse_00013'])
            self.assertTrue(report['replay_scope']['same_source_and_context'])
            self.assertEqual(report['replay_scope']['context_origins'],
                             {'coarse_00013': 'recorded_request'})
            self.assertEqual(report['results'][0]['context_origin'], 'recorded_request')
            self.assertFalse(report['replay_scope']['byte_identical_payload'])


if __name__ == '__main__':
    unittest.main()
