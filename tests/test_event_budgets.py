"""Regression cases for source starvation and coarse work consuming review time."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from volleymole.common import APP, read_json, save_json
from volleymole.event_pipeline import Discovery, analysis_summary, complete_collections
from volleymole.match_collection import execute_match
from volleymole.run_match import argument_parser, validate_event_arguments


class Clock:
    def __init__(self):
        self.value = 1000.
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.value

    def advance(self, amount):
        with self.lock:
            self.value += amount


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.args = argument_parser().parse_args(['--video', 'unused', '--model', 'fixture',
            '--no-sound-model', '--analysis-timeout', '100', '--semantic-concurrency', '1',
            '--analysis-cache-dir', str(self.root/'cache'), '--collection', 'both'])
        self.source = {'duration_sec': 50., 'identity': {'sha256': 'fixture'}, 'has_audio': False}
        self.manifest = {'source': self.source, 'config': read_json(APP/'defaults.json'),
            'rallies': [{'rally_id': 'local', 'start_sec': 10., 'end_sec': 15.,
                         'safe_start_sec': 9., 'safe_end_sec': 16.}]}
        for mock in (patch('volleymole.event_pipeline.time', self.clock),
                     patch('volleymole.semantic.time', self.clock),
                     patch.dict(os.environ, {'VOLLEYMOLE_API_KEY': 'fixture'}),
                     patch('volleymole.audio_events.cached_audio', return_value=(None,
                         {'status': 'unknown', 'windows': []}))):
            mock.start()
            self.addCleanup(mock.stop)

    def result(self, job, local):
        return {'job': job, 'events': [], 'signature': job['id'], 'local': local,
                'requested_fps': 2 if job['phase'] == 'coarse' else 8,
                'sample_times_sec': [], 'request': {}, 'cached': False, 'elapsed_sec': 1.}

    def exhausted_coarse(self, job, source, cache, settings, audio, local, deadline, records=None):
        if job['phase'] == 'coarse':
            self.clock.advance(deadline-self.clock.monotonic())
            raise TimeoutError('coarse timed out')
        self.assertGreater(deadline, self.clock.monotonic())
        self.clock.advance(1.)
        return self.result(job, local)

    def test_exhausted_coarse_still_reviews_local_rallies_with_reserved_budget(self):
        with patch('volleymole.event_pipeline.understand_context', side_effect=self.exhausted_coarse) as request:
            report = Discovery(self.source, self.root, self.args).finish(self.manifest)
        self.assertEqual([call.args[0]['phase'] for call in request.call_args_list], ['coarse', 'review'])
        self.assertEqual(report['coarse_completed'], 0)
        self.assertEqual(report['review_completed'], 1)
        self.assertEqual(report['phase_timing']['review_budget_sec'], 40.)
        self.assertTrue(report['phase_timing']['coarse_deadline_reached'])
        self.assertEqual(report['analysis_status'], 'partial')

    def test_slow_local_inference_does_not_spend_reserved_review_time(self):
        with patch('volleymole.event_pipeline.understand_context', side_effect=self.exhausted_coarse):
            discovery = Discovery(self.source, self.root, self.args)
            discovery.future.result(timeout=5)
            self.clock.advance(200.)  # Manifest arrives after the original total deadline.
            report = discovery.finish(self.manifest)
        self.assertEqual(report['review_completed'], 1)
        self.assertEqual(report['phase_timing']['local_manifest_wait_sec'], 200.)
        self.assertEqual(report['phase_timing']['review_budget_sec'], 40.)

    def test_unused_coarse_time_transfers_to_review_without_charging_manifest_wait(self):
        source = {**self.source, 'duration_sec': 20.}
        def response(job, source, cache, settings, audio, local, deadline, records=None):
            self.clock.advance(10. if job['phase'] == 'coarse' else 1.)
            return self.result(job, local)
        with patch('volleymole.event_pipeline.understand_context', side_effect=response):
            discovery = Discovery(source, self.root, self.args)
            discovery.future.result(timeout=5)
            self.clock.advance(20.)
            report = discovery.finish(self.manifest)
        self.assertEqual(report['phase_timing']['review_budget_sec'], 90.)
        self.assertEqual(report['phase_timing']['local_manifest_wait_sec'], 20.)
        self.assertEqual(report['analysis_status'], 'complete')

    def test_review_itself_stops_at_its_own_deadline(self):
        def response(job, source, cache, settings, audio, local, deadline, records=None):
            if job['phase'] == 'coarse':
                self.clock.advance(1.)
                return self.result(job, local)
            self.clock.advance(deadline-self.clock.monotonic())
            raise TimeoutError('review timed out')
        manifest = {**self.manifest, 'rallies': self.manifest['rallies']+[
            {**self.manifest['rallies'][0], 'rally_id': 'second'}]}
        with patch('volleymole.event_pipeline.understand_context', side_effect=response) as request:
            report = Discovery(self.source, self.root, self.args).finish(manifest)
        self.assertEqual(sum(c.args[0]['phase'] == 'review' for c in request.call_args_list), 1)
        self.assertEqual(report['review_completed'], 0)
        self.assertTrue(report['deadline_reached'])
        self.assertTrue(any(f['job'] == 'second' and f['error'] == 'deadline_or_disabled' for f in report['failures']))

    def test_later_sets_get_fresh_budgets_through_real_match_orchestration(self):
        self.args.output = self.root/'matches'
        self.args.llm_config = self.root/'absent.json'
        self.args.stop_after = 'render'
        def local_analysis(command, log):
            part = Path(command[command.index('--output')+1])
            save_json(part/'match_manifest.json', {**self.manifest, 'rallies': []})
        remaining_budgets = []
        def response(job, source, cache, settings, audio, local, deadline, records=None):
            remaining_budgets.append(deadline-self.clock.monotonic())
            self.clock.advance(deadline-self.clock.monotonic())
            raise TimeoutError('first coarse request uses this source budget')
        with patch('volleymole.common.probe', return_value=self.source), \
             patch('volleymole.common.identity', return_value=self.source['identity']), \
             patch('volleymole.match_collection.run', side_effect=local_analysis), \
             patch('volleymole.match_collection.time', self.clock), \
             patch('volleymole.event_pipeline.understand_context', side_effect=response) as requests, \
             patch('volleymole.media_worker.render') as render:
            result = execute_match(self.args, '2026-01-06', [(i, self.root/f'{i}.mp4') for i in (1, 2, 3)])
        # Old shared date deadline starved set 2 and 3 after set 1 spent it.
        self.assertEqual(requests.call_count, 3)
        self.assertEqual(remaining_budgets, [60., 60., 60.])
        self.assertEqual(result['status'], 'analysis_incomplete')
        self.assertEqual(result['analysis']['status'], 'failed')
        self.assertEqual(result['analysis']['source_count'], 3)
        render.assert_not_called()
        for collection in result['collections']:
            self.assertIsNone(collection['output'])
            self.assertIn('事件分析未完成', collection['shortage_reason'])

    def test_sound_evidence_survives_remote_failure_for_local_review(self):
        self.args.sound_model = self.root/'sound.pt'
        self.args.sound_labels = self.root/'labels.json'
        self.args.sound_model.write_bytes(b'fixture')
        save_json(self.args.sound_labels, ['Laughter'])
        sound = {'label': 'Laughter', 'start_sec': 10., 'end_sec': 11., 'probability': .9}
        def response(job, source, cache, settings, audio, local, deadline, records=None):
            if job['phase'] == 'review': self.assertIn(sound, local['sound_events'])
            return self.exhausted_coarse(job, source, cache, settings, audio, local, deadline, records)
        with patch('volleymole.audio_events.cached_audio', return_value=(np.zeros(32000*50),
                {'status': 'measured', 'windows': []})), \
             patch('volleymole.audio_events.LocalSoundDetector') as detector, \
             patch('volleymole.event_pipeline.understand_context', side_effect=response):
            detector.return_value.detect.return_value = [sound]
            report = Discovery(self.source, self.root, self.args).finish(self.manifest)
        self.assertEqual(report['review_completed'], 1)

    def test_complete_empty_match_and_failed_empty_match_are_distinct(self):
        timeline = {'events': [], 'coarse_status': 'complete', 'coarse_chunks': 3,
                    'coarse_completed': 3, 'review_candidates': 1, 'review_completed': 1, 'failures': []}
        self.assertEqual(analysis_summary(timeline)['status'], 'complete')
        self.assertEqual(analysis_summary({**timeline, 'coarse_status': 'partial',
            'failures': [{'error': 'TimeoutError'}]})['status'], 'partial')
        self.assertEqual(analysis_summary({**timeline, 'review_completed': 0})['status'], 'partial')
        reports = complete_collections(self.args, {**self.manifest, 'rallies': []}, timeline, self.root)
        self.assertTrue(all(r['analysis_status'] == 'complete' for r in reports))
        self.assertNotIn('事件分析未完成', reports[0]['shortage_reason'])

    def test_budget_fraction_rejects_invalid_and_nonfinite_values(self):
        for value in ('0', '1', '-.1', 'nan', 'inf'):
            parser = argument_parser()
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                args = parser.parse_args(['--video', 'unused', '--no-sound-model', '--review-budget-fraction', value])
                validate_event_arguments(args, parser)


if __name__ == '__main__':
    unittest.main()
