"""Offline CLI routing tests; media/inference fixtures do not claim video quality."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from volleymole.common import APP, identity, read_json, save_json
from volleymole.ranker import rank, rule_decision
from volleymole.run_match import argument_parser, validate_event_arguments


def source(path):
    return {'path': str(path), 'identity': identity(path), 'duration_sec': 220.,
            'width': 320, 'height': 180, 'has_audio': False, 'start_sec': 0.,
            'video_start_sec': 0., 'rotation': 0, 'frame_count': 6600,
            'nominal_fps': '30/1', 'average_fps': '30/1'}


def fixture_manifest(directory, video, count=10, score_offset=0, selection_mode=None):
    rallies = []
    for index in range(count):
        tracking = f'tracking/tracks/r{index}.json'
        save_json(directory/tracking, {'samples': []})
        previews = [f'previews/r{index}_{label}.jpg' for label in ('start', 'peak', 'end')]
        for name in previews:
            path = directory/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'local preview fixture; never uploaded')
        rallies.append({'rally_id': f'r{index}', 'start_sec': index*20.+2,
            'end_sec': index*20.+10, 'safe_start_sec': index*20.,
            'safe_end_sec': index*20.+12, 'duration_sec': 8., 'eligible': True,
            'rule_score': 100+score_offset-index, 'tracking_json': tracking,
            'preview_frames': previews, 'preview_times_sec': [index*20.+2, index*20.+6, index*20.+10],
            'actions': [], 'players': [], 'ball_metrics': {'visible_ratio': .8, 'trajectory_changes': 5},
            'exclusion_reasons': []})
    manifest = {'source': source(video), 'config': read_json(APP/'defaults.json'), 'rallies': rallies}
    if selection_mode is not None:
        manifest['selection_mode'] = selection_mode
    save_json(directory/'match_manifest.json', manifest)
    return manifest


def fixture_previews(directory, ids, *unused):
    manifest = read_json(directory/'match_manifest.json')
    files = [name for rally in manifest['rallies'] if rally['rally_id'] in ids
             for name in rally['preview_frames']]
    save_json(directory/'previews/index.json', {'files': files})


def fixture_render(directory, *unused):
    output = directory/'top10.mp4'; output.write_bytes(b'fake rendered media for route testing')
    save_json(directory/'render_report.json', {'output': str(output), 'expected_duration_sec': 120.,
                                             'clips': [], 'segments': []})


def fixture_verify(directory, top_k, style, alignment_python):
    path = directory/'verification.json'
    save_json(path, {'status': 'fixture', 'top_k': top_k, 'alignment_python': alignment_python})
    return str(path), [path]


def valid_api_decision(candidates, directory, top_k, *unused, **kwargs):
    return rule_decision(candidates, top_k)


def empty_timeline():
    return {'events': [], 'coarse_status': 'complete', 'coarse_chunks': 0, 'coarse_completed': 0,
            'review_candidates': 0, 'review_completed': 0, 'failures': []}


class AnalysisModeArgumentTests(unittest.TestCase):
    def parse(self, *options):
        parser = argument_parser()
        args = parser.parse_args(['--video', 'unused.mp4', '--no-sound-model', *options])
        validate_event_arguments(args, parser)
        return args

    def test_auto_resolves_highlights_to_rallies_and_dual_or_funny_to_events(self):
        self.assertEqual(argument_parser().get_default('analysis_mode'), 'auto')
        self.assertEqual(self.parse().analysis_mode, 'rallies')
        for collection in ('bloopers', 'both'):
            with self.subTest(collection=collection):
                self.assertEqual(self.parse('--collection', collection).analysis_mode, 'events')

    def test_explicit_modes_and_rules_highlights_are_preserved(self):
        for mode in ('rallies', 'events'):
            with self.subTest(mode=mode):
                self.assertEqual(self.parse('--analysis-mode', mode).analysis_mode, mode)
        self.assertEqual(self.parse('--ranker', 'rules').analysis_mode, 'rallies')
        self.assertEqual(self.parse('--ranker', 'rules', '--analysis-mode', 'rallies').analysis_mode, 'rallies')

    def test_invalid_mode_collection_and_ranker_combinations_fail(self):
        invalid = [('--analysis-mode', 'rallies', '--collection', kind) for kind in ('bloopers', 'both')]
        invalid += [('--ranker', 'rules', '--collection', kind) for kind in ('bloopers', 'both')]
        invalid.append(('--ranker', 'rules', '--analysis-mode', 'events'))
        for options in invalid:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                self.parse(*options)
            self.assertEqual(exc.exception.code, 2)

    def test_events_rules_can_build_local_manifest_only(self):
        args = self.parse('--analysis-mode', 'events', '--ranker', 'rules', '--stop-after', 'manifest')
        self.assertEqual(args.analysis_mode, 'events')


class AnalysisRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Rendering is mocked; only stable bytes for its font fingerprint matter.
        self.font = self.root/'font.fixture'
        self.font.write_bytes(b'font fingerprint fixture, not a renderable font')
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.art_assets = self.stack.enter_context(patch('volleymole.illustrated.asset_paths',
            side_effect=AssertionError('Classic routing must not read lively artwork')))
        self.stack.enter_context(patch.dict(os.environ, {'VOLLEYMOLE_API_KEY': 'local-test-key',
            'OPENAI_API_KEY': '', 'VOLLEYMOLE_MODEL': '', 'VOLLEYMOLE_VISION_MODEL': ''}))
        self.stack.enter_context(patch('volleymole.semantic.request_json',
                                     side_effect=AssertionError('Network requests are forbidden in routing tests')))
        self.api = self.stack.enter_context(patch('volleymole.ranker.api_decision', side_effect=valid_api_decision))
        self.discovery = self.stack.enter_context(patch('volleymole.event_pipeline.Discovery'))
        self.discovery.return_value.finish.return_value = empty_timeline()
        self.collections = self.stack.enter_context(patch('volleymole.event_pipeline.complete_collections', return_value=[]))
        self.build_modes = []
        self.part_modes = []

    def common_options(self):
        return ['--top-k', '10', '--style', 'classic', '--no-sound-model', '--model', 'glm-5.3-flash',
                '--font', str(self.font),
                '--llm-config', str(self.root/'absent-config.json'),
                '--analysis-cache-dir', str(self.root/'cache')]

    def run_single(self, *options):
        from volleymole.run_match import main
        video = self.root/'source.mp4'; video.write_bytes(b'source fixture')
        output = self.root/'single'
        def ingest(video, directory, registry, signature, cache, force, frame_cache, fps):
            paths = {'analytics': directory/'analytics/detections.jsonl',
                     'tracking': directory/'tracking/ball.csv', 'player': directory/'player/index.json'}
            artifacts = []
            for name, path in paths.items():
                save_json(directory/name/'provenance.json', {'mode': 'fixture'})
                path.write_text(''); artifacts.extend([path, directory/name/'provenance.json'])
            pts = directory/'tracking/source_pts.csv'; pts.write_text('0\n'); artifacts.append(pts)
            if frame_cache: save_json(frame_cache/'status.json', {'status': 'complete'})
            return {k: str(v) for k, v in paths.items()}, artifacts
        def build(*params, **kwargs):
            directory = params[5]
            self.assertIn(kwargs['selection_mode'], ('rallies', 'events'))
            self.build_modes.append(kwargs['selection_mode'])
            manifest = fixture_manifest(directory, video, selection_mode=kwargs['selection_mode'])
            return manifest, [directory/'match_manifest.json']
        def media_command(command, *unused):
            command = [str(x) for x in command]
            if 'previews' in command:
                fixture_previews(output, command[command.index('--rally-ids')+1:])
            elif 'render' in command:
                fixture_render(output)
            else: raise AssertionError('Unexpected external command')
        self.stack.enter_context(patch('volleymole.run_match.ModelRegistry',
                                     return_value=SimpleNamespace(directory=self.root, entries={})))
        self.stack.enter_context(patch('volleymole.run_match.probe', side_effect=source))
        self.stack.enter_context(patch('volleymole.run_match.inference_signature', return_value={'fixture': True}))
        self.stack.enter_context(patch('volleymole.run_match.ingest_shared', side_effect=ingest))
        self.stack.enter_context(patch('volleymole.run_match.build_manifest', side_effect=build))
        self.stack.enter_context(patch('volleymole.run_match.run', side_effect=media_command))
        rank_call = self.stack.enter_context(patch('volleymole.run_match.rank', wraps=rank))
        verify_call = self.stack.enter_context(patch('volleymole.run_match.verify', side_effect=fixture_verify))
        main(['--video', str(video), '--output', str(output), *self.common_options(), *options])
        return output, rank_call, verify_call

    def run_match(self, *options):
        from volleymole.match_collection import main
        videos = self.root/'videos'; videos.mkdir(exist_ok=True)
        for number in (1, 2): (videos/f'2026.1.6.{number}.mp4').write_bytes(b'source fixture')
        root_output = self.root/'matches'
        output = root_output/'2026-01-06-top10'
        def local_analysis(command, *unused):
            command = [str(x) for x in command]
            self.assertEqual(command[:3], [sys.executable, '-m', 'volleymole.run_match'])
            self.assertIn('--analysis-mode', command)
            parser = argument_parser()
            child = parser.parse_args(command[3:])
            validate_event_arguments(child, parser)
            self.assertEqual((child.ranker, child.stop_after), ('rules', 'manifest'))
            self.part_modes.append(child.analysis_mode)
            part = Path(command[command.index('--output')+1]); video = Path(command[command.index('--video')+1])
            fixture_manifest(part, video, count=6, score_offset=int(video.stem.split('.')[-1])*100,
                             selection_mode=child.analysis_mode)
        self.stack.enter_context(patch('volleymole.match_collection.run', side_effect=local_analysis))
        self.stack.enter_context(patch('volleymole.common.probe', side_effect=source))
        self.stack.enter_context(patch('volleymole.media_worker.previews', side_effect=fixture_previews))
        self.stack.enter_context(patch('volleymole.media_worker.render', side_effect=fixture_render))
        rank_call = self.stack.enter_context(patch('volleymole.ranker.rank', wraps=rank))
        verify_call = self.stack.enter_context(patch('volleymole.match_collection.verify', side_effect=fixture_verify))
        main(['--input-dir', str(videos), '--output', str(root_output), *self.common_options(), *options])
        return output, rank_call, verify_call

    def assert_rally_run(self, output, rank_call, verify_call, *, fallback=False):
        self.discovery.assert_not_called(); self.collections.assert_not_called()
        self.art_assets.assert_not_called()
        rank_call.assert_called_once()
        verify_call.assert_called_once_with(output, 10, 'classic', sys.executable)
        config = read_json(output/'run_config.json')
        self.assertEqual(config['analysis_mode'], 'rallies')
        self.assertEqual(config['collection'], 'highlights')
        self.assertEqual(read_json(output/'match_manifest.json')['selection_mode'], 'rallies')
        decision = read_json(output/'edit_decision.json')
        self.assertEqual(len(decision['selected']), 10)
        self.assertEqual(decision['ranking_mode'], 'rules_fallback' if fallback else 'multimodal_api')
        self.assertFalse((output/'event_timeline.json').exists())
        self.assertEqual(read_json(output/'state.json')['stages']['verify']['status'], 'complete')

    def test_default_single_auto_reaches_legacy_rank_and_alignment_verification(self):
        self.assert_rally_run(*self.run_single())
        self.api.assert_called_once()
        self.assertEqual(self.build_modes, ['rallies'])

    def test_default_multiset_auto_reaches_global_top10_and_alignment_verification(self):
        result = self.run_match(); self.assert_rally_run(*result)
        manifest = read_json(result[0]/'match_manifest.json')
        self.assertEqual(len(manifest['sources']), 2)
        self.assertEqual(len(manifest['rallies']), 12)
        self.assertEqual(self.part_modes, ['rallies', 'rallies'])
        selected = read_json(result[0]/'selected_sources.json')['selected']
        self.assertEqual(sum(item['set_number'] == 2 for item in selected), 6)

    def test_default_single_model_failure_still_uses_strict_rally_fallback(self):
        self.api.side_effect = OSError('simulated offline transport')
        self.assert_rally_run(*self.run_single(), fallback=True)

    def test_default_multiset_model_failure_still_uses_strict_rally_fallback(self):
        self.api.side_effect = OSError('simulated offline transport')
        self.assert_rally_run(*self.run_match(), fallback=True)

    def test_explicit_single_events_bypasses_legacy_rank(self):
        output, rank_call, verify_call = self.run_single('--analysis-mode', 'events')
        self.discovery.assert_called_once(); self.discovery.return_value.finish.assert_called_once()
        self.collections.assert_called_once(); rank_call.assert_not_called(); verify_call.assert_not_called()
        self.assertEqual(read_json(output/'run_config.json')['analysis_mode'], 'events')
        self.assertEqual(read_json(output/'match_manifest.json')['selection_mode'], 'events')
        self.assertEqual(self.build_modes, ['events'])

    def test_explicit_multiset_events_bypasses_legacy_rank(self):
        output, rank_call, verify_call = self.run_match('--analysis-mode', 'events')
        self.assertEqual(self.discovery.call_count, 2)
        self.assertEqual(self.discovery.return_value.finish.call_count, 2)
        self.collections.assert_called_once(); rank_call.assert_not_called(); verify_call.assert_not_called()
        self.assertEqual(read_json(output/'run_config.json')['analysis_mode'], 'events')
        self.assertEqual(read_json(output/'match_manifest.json')['selection_mode'], 'events')
        self.assertEqual(self.part_modes, ['events', 'events'])
        self.assertTrue((output/'event_timeline.json').is_file())

    def test_default_multiset_both_still_uses_event_collections(self):
        output, rank_call, verify_call = self.run_match('--collection', 'both')
        self.assertEqual(self.discovery.call_count, 2)
        self.collections.assert_called_once(); rank_call.assert_not_called(); verify_call.assert_not_called()
        config = read_json(output/'run_config.json')
        self.assertEqual((config['analysis_mode'], config['collection']), ('events', 'both'))
        self.assertEqual(self.part_modes, ['events', 'events'])

    def test_default_single_bloopers_still_uses_event_collections(self):
        output, rank_call, verify_call = self.run_single('--collection', 'bloopers')
        self.discovery.assert_called_once(); self.collections.assert_called_once()
        rank_call.assert_not_called(); verify_call.assert_not_called()
        config = read_json(output/'run_config.json')
        self.assertEqual((config['analysis_mode'], config['collection']), ('events', 'bloopers'))

    def test_single_rank_stop_does_not_render_or_verify(self):
        output, rank_call, verify_call = self.run_single('--stop-after', 'rank')
        rank_call.assert_called_once(); verify_call.assert_not_called(); self.discovery.assert_not_called()
        self.assertEqual(len(read_json(output/'edit_decision.json')['selected']), 10)
        self.assertFalse((output/'render_report.json').exists())

    def test_single_replay_stage_runs_after_rank_and_before_render(self):
        def review(directory,args):
            self.assertEqual(len(read_json(directory/'edit_decision.json')['selected']),10)
            self.assertFalse((directory/'render_report.json').exists())
            return {'status':'complete'}
        with patch('volleymole.replay_stage.run_review',side_effect=review) as called:
            output,rank_call,verify_call=self.run_single('--stop-after','replay')
        called.assert_called_once();rank_call.assert_called_once();verify_call.assert_not_called()
        self.assertFalse((output/'render_report.json').exists())

    def test_multiset_replay_stage_runs_after_global_ranking(self):
        def review(directory,args):
            self.assertEqual(len(read_json(directory/'match_manifest.json')['sources']),2)
            self.assertEqual(len(read_json(directory/'edit_decision.json')['selected']),10)
            self.assertFalse((directory/'render_report.json').exists())
            return {'status':'complete'}
        with patch('volleymole.replay_stage.run_review',side_effect=review) as called:
            output,rank_call,verify_call=self.run_match('--stop-after','replay')
        called.assert_called_once();rank_call.assert_called_once();verify_call.assert_not_called()
        self.assertEqual(read_json(output/'match_summary.json')['replay_review_status'],'complete')

    def test_required_replay_failure_prevents_render_and_verify(self):
        with patch('volleymole.replay_stage.run_review',side_effect=RuntimeError('review incomplete')), \
                self.assertRaisesRegex(RuntimeError,'review incomplete'):
            self.run_single('--replay-review','required')
        self.assertFalse((self.root/'single/render_report.json').exists())
        self.assertNotIn('render',read_json(self.root/'single/state.json')['stages'])

    def test_manifest_stop_never_starts_event_analysis(self):
        output, rank_call, verify_call = self.run_single('--analysis-mode', 'events', '--stop-after', 'manifest')
        self.discovery.assert_not_called(); self.collections.assert_not_called()
        rank_call.assert_not_called(); verify_call.assert_not_called()
        self.assertTrue((output/'match_manifest.json').is_file())
        self.assertEqual(read_json(output/'run_config.json')['analysis_mode'], 'events')

    def test_switching_mode_rebuilds_manifest_but_same_mode_reuses_it(self):
        output, _, _ = self.run_single('--stop-after', 'manifest')
        rally_signature = read_json(output/'state.json')['stages']['rallies']['signature']
        self.assertEqual(self.build_modes, ['rallies'])
        self.assertEqual(read_json(output/'match_manifest.json')['selection_mode'], 'rallies')

        self.run_single('--analysis-mode', 'events', '--stop-after', 'manifest')
        event_signature = read_json(output/'state.json')['stages']['rallies']['signature']
        self.assertNotEqual(rally_signature, event_signature)
        self.assertEqual(self.build_modes, ['rallies', 'events'])
        self.assertEqual(read_json(output/'match_manifest.json')['selection_mode'], 'events')

        self.run_single('--analysis-mode', 'events', '--stop-after', 'manifest')
        self.assertEqual(self.build_modes, ['rallies', 'events'])
        self.assertIn('rallies', read_json(output/'timing_latest.json')['reused_stages'])

        self.run_single('--stop-after', 'manifest')
        self.assertEqual(self.build_modes, ['rallies', 'events', 'rallies'])
        self.assertEqual(read_json(output/'state.json')['stages']['rallies']['signature'], rally_signature)
        self.assertEqual(read_json(output/'match_manifest.json')['selection_mode'], 'rallies')
        self.discovery.assert_not_called(); self.collections.assert_not_called(); self.api.assert_not_called()

    def seed_stale_match_summary(self):
        output = self.root/'matches/2026-01-06-top10'
        save_json(output/'match_summary.json', {
            'date': '2026-01-06', 'directory': str(output), 'status': 'analysis_incomplete',
            'analysis_mode': 'events', 'analysis': {'status': 'partial'},
            'collections': [{'collection': 'highlights', 'actual_count': 0,
                             'output': str(output/'stale-events.mp4')}],
            'output': str(output/'stale-events.mp4'),
            'ranking_mode': 'stale-event-ranking', 'fallback': {'type': 'stale_failure'},
        })

    def assert_fresh_early_match_summary(self, output, status):
        summary = read_json(output/'match_summary.json')
        self.assertEqual(summary['status'], status)
        self.assertEqual(summary['date'], '2026-01-06')
        self.assertEqual(summary['directory'], str(output))
        self.assertEqual(summary['analysis_mode'], 'rallies')
        self.assertNotIn('collections', summary)
        self.assertNotIn('analysis', summary)
        self.assertNotIn('output', summary)
        self.assertEqual(read_json(output.parent/'matches_report.json')['matches'], [summary])
        return summary

    def test_multiset_rank_stop_replaces_stale_event_summary(self):
        self.seed_stale_match_summary()
        output, rank_call, verify_call = self.run_match('--stop-after', 'rank')
        rank_call.assert_called_once(); verify_call.assert_not_called()
        self.discovery.assert_not_called(); self.collections.assert_not_called()
        self.assertFalse((output/'render_report.json').exists())
        summary = self.assert_fresh_early_match_summary(output, 'rank_only')
        self.assertEqual(summary['ranking_mode'], 'multimodal_api')
        self.assertIsNone(summary['fallback'])
        decision = read_json(output/'edit_decision.json')
        self.assertEqual(summary['ranking_mode'], decision['ranking_mode'])
        self.assertEqual(summary['fallback'], decision['fallback'])

    def test_multiset_rank_stop_reports_current_fallback_not_stale_events(self):
        self.seed_stale_match_summary()
        self.api.side_effect = OSError('simulated offline transport')
        output, rank_call, verify_call = self.run_match('--stop-after', 'rank')
        rank_call.assert_called_once(); verify_call.assert_not_called()
        self.discovery.assert_not_called(); self.collections.assert_not_called()
        summary = self.assert_fresh_early_match_summary(output, 'rank_only')
        self.assertEqual(summary['ranking_mode'], 'rules_fallback')
        self.assertEqual(summary['fallback']['type'], 'OSError')
        self.assertEqual(summary['fallback'], read_json(output/'edit_decision.json')['fallback'])

    def test_multiset_manifest_stop_replaces_stale_event_summary(self):
        self.seed_stale_match_summary()
        output, rank_call, verify_call = self.run_match('--stop-after', 'manifest')
        rank_call.assert_not_called(); verify_call.assert_not_called()
        self.api.assert_not_called(); self.discovery.assert_not_called(); self.collections.assert_not_called()
        self.assertTrue((output/'match_manifest.json').is_file())
        summary = self.assert_fresh_early_match_summary(output, 'manifest_only')
        self.assertNotIn('ranking_mode', summary)
        self.assertNotIn('fallback', summary)


class AnalysisManifestMergeTests(unittest.TestCase):
    def make_parts(self, root, modes):
        parts = []
        for number, mode in enumerate(modes, 1):
            video = root/f'source-{number}.mp4'; video.write_bytes(b'local source fixture')
            part = root/'parts'/f'set-{number:04d}'
            fixture_manifest(part, video, count=6, selection_mode=mode)
            parts.append((number, part))
        return parts

    def test_merging_same_mode_preserves_selection_mode(self):
        from volleymole.match_collection import merge_manifests
        for mode in ('rallies', 'events'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = merge_manifests(self.make_parts(root, [mode, mode]), root, '2026-01-06')
                self.assertEqual(manifest['selection_mode'], mode)
                self.assertEqual(len(manifest['sources']), 2)
                self.assertEqual(len(manifest['rallies']), 12)

    def test_merging_different_modes_is_rejected(self):
        from volleymole.match_collection import merge_manifests
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = self.make_parts(root, ['rallies', 'events'])
            with self.assertRaisesRegex(ValueError, '选择模式不一致'):
                merge_manifests(parts, root, '2026-01-06')


if __name__ == '__main__':
    unittest.main()
