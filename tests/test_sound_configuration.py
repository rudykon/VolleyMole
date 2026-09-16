"""Installed sound assets are discovered predictably and cache device changes."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from volleymole.run_match import argument_parser, validate_event_arguments
from volleymole.event_pipeline import Discovery
from volleymole.common import digest


class SoundConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.addCleanup(patch.stopall)
        patch('volleymole.run_match.ROOT', self.root).start()
        patch.dict(os.environ, {'VOLLEYMOLE_MODELS': str(self.root/'models')}).start()

    def assets(self, directory):
        directory = directory/'audio'
        directory.mkdir(parents=True)
        model = directory/'panns_cnn14_sed.pt'
        labels = directory/'panns_cnn14_sed.labels.json'
        model.write_bytes(b'configuration-only model placeholder')
        labels.write_text('["Laughter"]')
        return model, labels

    def arguments(self, *extra):
        parser = argument_parser()
        args = parser.parse_args(['--video', 'fixture.mp4', '--analysis-mode', 'events', *map(str, extra)])
        with contextlib.redirect_stderr(io.StringIO()):
            validate_event_arguments(args, parser)
        return args

    def sidecar(self, model, labels, threshold=.2, path=None, **overrides):
        path = path or model.with_suffix('.thresholds.json')
        data = {'schema_version': 1, 'model_sha256': digest(model),
                'labels_sha256': digest(labels), 'thresholds': {'Laughter': threshold}, **overrides}
        path.write_text(json.dumps(data))
        return path

    def test_default_model_directory_is_auto_discovered(self):
        expected = self.assets(self.root/'models')
        with patch.dict(os.environ):
            os.environ.pop('VOLLEYMOLE_MODELS', None)
            args = self.arguments()
        self.assertEqual((args.sound_model, args.sound_labels), expected)

    def test_explicit_model_directory_precedes_environment(self):
        self.assets(self.root/'models')
        expected = self.assets(self.root/'custom')
        args = self.arguments('--models', self.root/'custom')
        self.assertEqual((args.sound_model, args.sound_labels), expected)

    def test_environment_model_directory_is_supported(self):
        expected = self.assets(self.root/'environment')
        with patch.dict(os.environ, {'VOLLEYMOLE_MODELS': str(self.root/'environment')}):
            args = self.arguments()
        self.assertEqual((args.sound_model, args.sound_labels), expected)

    def test_disable_and_rules_mode_do_not_enable_installed_model(self):
        self.assets(self.root/'models')
        for extra in (('--no-sound-model',), ('--ranker', 'rules', '--analysis-mode', 'rallies')):
            with self.subTest(extra=extra):
                args = self.arguments(*extra)
                self.assertIsNone(args.sound_model)
                self.assertIsNone(args.sound_labels)

    def test_default_rally_route_does_not_load_event_sound_model(self):
        self.assets(self.root/'models')
        parser = argument_parser()
        args = parser.parse_args(['--video', 'fixture.mp4'])
        validate_event_arguments(args, parser)
        self.assertEqual(args.analysis_mode, 'rallies')
        self.assertIsNone(args.sound_model)
        self.assertIsNone(args.sound_labels)

    def test_partial_install_is_not_auto_enabled(self):
        model, labels = self.assets(self.root/'models')
        labels.unlink()
        args = self.arguments()
        self.assertIsNone(args.sound_model)

    def test_explicit_pair_is_used_and_incomplete_or_missing_paths_fail(self):
        model, labels = self.assets(self.root/'explicit')
        args = self.arguments('--sound-model', model, '--sound-labels', labels)
        self.assertEqual((args.sound_model, args.sound_labels), (model, labels))
        for extra in (('--sound-model', model), ('--sound-labels', labels),
                      ('--sound-model', model, '--sound-labels', self.root/'missing.json'),
                      ('--sound-model', model, '--sound-labels', labels, '--no-sound-model')):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.arguments(*extra)

    def test_bound_threshold_sidecar_is_discovered_or_explicitly_selected(self):
        model, labels = self.assets(self.root/'models')
        auto = self.sidecar(model, labels)
        args = self.arguments()
        self.assertEqual(args.sound_thresholds, auto)
        explicit = self.sidecar(model, labels, .3, self.root/'custom-thresholds.json')
        args = self.arguments('--sound-thresholds', explicit)
        self.assertEqual(args.sound_thresholds, explicit)
        self.assertIsNone(self.arguments('--no-sound-model').sound_thresholds)

    def test_foreign_threshold_sidecar_is_never_guessed(self):
        model, labels = self.assets(self.root/'models')
        self.sidecar(model, labels, path=model.parent/'another_model.thresholds.json')
        self.assertIsNone(self.arguments().sound_thresholds)

    def test_thresholds_require_matching_model_labels_and_valid_classes(self):
        model, labels = self.assets(self.root/'models')
        for overrides in ({'model_sha256': 'foreign-model'}, {'labels_sha256': 'foreign-label-order'},
                          {'schema_version': 2}, {'thresholds': {'Laughter': True}},
                          {'thresholds': {'Laughter': float('nan')}}, {'thresholds': {'Invented': .2}},
                          {'thresholds': {'Laughter': 1.1}}):
            with self.subTest(overrides=overrides):
                self.sidecar(model, labels, **overrides)
                with self.assertRaises(SystemExit):
                    self.arguments()
        model.with_suffix('.thresholds.json').unlink()
        labels.write_text('["Speech"]')
        with self.assertRaises(SystemExit):
            self.arguments('--sound-thresholds', self.root/'missing.json')

    def test_thresholds_cannot_be_used_without_or_with_disabled_detector(self):
        path = self.root/'thresholds.json'
        path.write_text('{}')
        with self.assertRaises(SystemExit): self.arguments('--sound-thresholds', path)
        with self.assertRaises(SystemExit): self.arguments('--no-sound-model', '--sound-thresholds', path)

    def test_sound_cache_reuses_same_device_and_separates_other_devices(self):
        model, labels = self.assets(self.root/'explicit')
        source = {'identity': {'sha256': 'fixed-real-source-id'}, 'duration_sec': 1.}
        def understand(job, source, cache, settings, audio, local, deadline):
            return {'job': job, 'local': local, 'events': [], 'signature': 'test'}
        with patch.dict(os.environ, {'VOLLEYMOLE_API_KEY': 'fixture-no-network'}), \
             patch('volleymole.audio_events.cached_audio', return_value=(np.zeros(32000, dtype=np.float32), {'windows': []})), \
             patch('volleymole.audio_events.LocalSoundDetector') as detector, \
             patch('volleymole.event_pipeline.understand_context', side_effect=understand):
            detector.return_value.detect.return_value = []
            for device in ('cpu', 'cpu', 'cuda:0'):
                args = self.arguments('--sound-model', model, '--sound-labels', labels,
                    '--model', 'fixture', '--sound-device', device,
                    '--analysis-cache-dir', self.root/'cache')
                discovery = Discovery(source, self.root/'run', args)
                try:
                    result = discovery.future.result(timeout=5)
                    self.assertEqual(result['status'], 'complete')
                finally:
                    discovery.close()
            self.assertEqual(detector.call_count, 2)
            self.assertEqual(detector.return_value.detect.call_count, 2)
            self.assertEqual([call.kwargs['device'] for call in detector.call_args_list], ['cpu', 'cuda:0'])

    def test_threshold_file_changes_invalidate_sound_cache(self):
        model, labels = self.assets(self.root/'models')
        source = {'identity': {'sha256': 'fixed-source'}, 'duration_sec': 1.}
        def understand(job, source, cache, settings, audio, local, deadline):
            return {'job': job, 'local': local, 'events': [], 'signature': 'test'}
        with patch.dict(os.environ, {'VOLLEYMOLE_API_KEY': 'fixture-no-network'}), \
             patch('volleymole.audio_events.cached_audio', return_value=(np.zeros(32000, dtype=np.float32), {'windows': []})), \
             patch('volleymole.audio_events.LocalSoundDetector') as detector, \
             patch('volleymole.event_pipeline.understand_context', side_effect=understand):
            detector.return_value.detect.return_value = []
            for threshold in (.2, .2, .3):
                self.sidecar(model, labels, threshold)
                args = self.arguments('--model', 'fixture', '--analysis-cache-dir', self.root/'cache')
                discovery = Discovery(source, self.root/'run', args)
                try: self.assertEqual(discovery.future.result(timeout=5)['status'], 'complete')
                finally: discovery.close()
            self.assertEqual(detector.call_count, 2)
            self.assertEqual(detector.return_value.detect.call_count, 2)
            self.assertEqual([call.kwargs['thresholds'] for call in detector.call_args_list],
                             [{'Laughter': .2}, {'Laughter': .3}])


if __name__ == '__main__':
    unittest.main()
