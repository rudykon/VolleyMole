"""Portable templates work through both editing entry points and local audio."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from volleymole.templates import BUILTINS, load_template, main, read_template, write_template
from volleymole.run_match import argument_parser


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = {'version': 1, 'name': 'my-team',
                         'options': {'design_suite': 'aurora', 'quality': '1440p'},
                         'audio': {'max_cues': 1, 'rms_db': -24}}
        self.path = write_template(self.document, self.root / 'input.json')

    def test_builtin_templates_use_valid_runtime_choices_without_binding_top_k(self):
        from volleymole.design_suites import resolve_design
        from volleymole.replay_stage import validate_arguments
        self.assertEqual(len(list(BUILTINS.glob('*.json'))), 5)
        for path in BUILTINS.glob('*.json'):
            parser = argument_parser()
            args = parser.parse_args(['--video', 'match.mp4', '--template', path.stem])
            self.assertEqual(args.top_k, 5)
            self.assertEqual(args.design_suite, path.stem)
            validate_arguments(args)
            self.assertEqual(resolve_design(args.design_suite, args.design_language)[1][-2:], 'zh')
            parser.set_defaults(top_k=10)
            self.assertEqual(parser.parse_args(['--video', 'match.mp4', '--template', path.stem]).top_k, 10)

    def test_import_switch_export_and_collision_leave_original_intact(self):
        library = self.root / 'library'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--directory', str(library), 'import', str(self.path)])
        with patch.dict(os.environ, {'VOLLEYMOLE_TEMPLATES': str(library)}):
            self.assertEqual(load_template('my-team'), self.document)
            for raw in (['--template', 'my-team', '--quality', '720p'],
                        ['--quality', '720p', '--template=my-team']):
                args = argument_parser().parse_args(['--video', 'match.mp4', *raw])
                self.assertEqual(args.design_suite, 'aurora')
                self.assertEqual(args.quality, '720p')
                self.assertEqual(args.template_snapshot['options']['quality'], '720p')
                self.assertEqual(args.template_snapshot['audio'], self.document['audio'])
        with self.assertRaises(FileExistsError):
            main(['--directory', str(library), 'import', str(self.path)])
        self.assertEqual(read_template(library / 'my-team.json'), self.document)
        exported = self.root / 'export.json'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--directory', str(library), 'export', 'my-team', '--output', str(exported)])
        self.assertEqual(read_template(exported), self.document)

    def test_parser_does_not_leak_template_defaults_into_next_invocation(self):
        parser = argument_parser()
        parser.parse_args(['--video', 'a.mp4', '--template', str(self.path)])
        plain = parser.parse_args(['--video', 'b.mp4'])
        self.assertEqual(plain.quality, '1080p')
        self.assertEqual(plain.design_suite, 'custom')
        self.assertIsNone(plain.template_snapshot)

    def test_replay_switches_flow_to_renderer_and_explicit_flags_win(self):
        from volleymole.media_worker import render
        from volleymole.font_support import DEFAULT_FONT
        from volleymole.replay_stage import validate_arguments, config_from
        self.document['options'].update(replays=False, replay_speed=.5)
        self.path.write_text(json.dumps(self.document))
        parser = argument_parser()
        args = parser.parse_args(['--video', 'match.mp4', '--template', str(self.path)])
        self.assertFalse(args.replays)
        self.assertEqual(args.replay_speed, .5)
        validate_arguments(args)
        args = parser.parse_args(['--video', 'match.mp4', '--template', str(self.path), '--replays'])
        self.assertTrue(args.replays)
        args = parser.parse_args(['--video', 'match.mp4', '--template', 'matchday', '--no-replays'])
        self.assertFalse(args.replays)
        (self.root / 'run_config.json').write_text(json.dumps(config_from(args)))
        with patch('volleymole.presentation.render_lively') as mocked:
            render(self.root, DEFAULT_FONT, 'lively')
            self.assertFalse(mocked.call_args.kwargs['replays'])
        args.replay_review = 'required'
        with self.assertRaises(ValueError):
            validate_arguments(args)

    def test_invalid_imports_never_create_library_files(self):
        bad_docs = [[], {**self.document, 'version': True},
                    {**self.document, 'name': '../escape'},
                    {**self.document, 'api_key': 'not-a-key'},
                    {**self.document, 'options': {'output': '/tmp/unsafe'}},
                    {**self.document, 'options': {'quality': '4k'}},
                    {**self.document, 'options': {'top_k': True}},
                    {**self.document, 'options': {'replay_review_timeout': float('nan')}},
                    {**self.document, 'options': {'replay_review_width': 2048}},
                    {**self.document, 'options': {'replay_scan_fps': 12, 'replay_review_fps': 4}},
                    {**self.document, 'audio': {'enabled': 'false'}},
                    {**self.document, 'audio': {'rms_db': 0}},
                    {**self.document, 'audio': {'max_cues': False}},
                    {**self.document, 'audio': {'script': 'anything'}}]
        library = self.root / 'library'
        for document in bad_docs:
            self.path.write_text(json.dumps(document))
            with self.subTest(document=document), self.assertRaises(ValueError):
                main(['--directory', str(library), 'import', str(self.path)])
            self.assertFalse(library.exists())
        self.path.write_text('{"version": 1, "version": 2}')
        with self.assertRaisesRegex(ValueError, '重复字段'):
            read_template(self.path)
        self.path.write_text(' ' * (64 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, '64 KiB'):
            read_template(self.path)

    def test_invalid_template_fails_before_model_setup(self):
        from volleymole.run_match import main as run
        with patch('volleymole.run_match.ModelRegistry') as registry, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run(['--video', 'a.mp4', '--template', str(self.root / 'missing.json')])
            registry.assert_not_called()

    def test_match_entry_receives_template_and_explicit_overrides(self):
        from volleymole.match_collection import main as match
        video = self.root / '2026.9.15.1.mp4'
        video.touch()
        output = self.root / 'output'
        with patch('volleymole.match_collection.execute_match', return_value={'status': 'tested'}) as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                match(['--input-dir', str(self.root), '--output', str(output), '--ranker', 'rules',
                       '--template', str(self.path), '--quality', '720p'])
        args, day, sets = execute.call_args.args
        self.assertEqual((args.top_k, args.quality, args.design_suite), (10, '720p', 'aurora'))
        self.assertEqual(day, '2026-09-15')
        self.assertEqual(sets, [(1, video)])

    def test_export_run_omits_credentials_paths_and_preserves_audio(self):
        config = {'video': '/private/match.mp4', 'api_key': 'private-value', 'models': '/private/models',
                  'quality': '2160p', 'style': 'lively', 'design_suite': 'sumi',
                  'template': self.document, 'focus_player': None}
        (self.root / 'run_config.json').write_text(json.dumps(config))
        target = self.root / 'export.json'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['export', '--from-run', str(self.root), '--output', str(target)])
        result = read_template(target)
        self.assertEqual(result['options'], {'quality': '2160p', 'style': 'lively', 'design_suite': 'sumi'})
        self.assertEqual(result['audio'], self.document['audio'])
        self.assertNotIn('private', target.read_text())

    def test_audio_cli_applies_same_template_and_final_overrides(self):
        from volleymole.meme_audio import main as audio, apply_audio_settings
        with patch('volleymole.meme_audio.render', return_value={
            'output': 'out.mp4', 'cue_count': 1, 'video_unchanged': True}) as render:
            with contextlib.redirect_stdout(io.StringIO()):
                audio(['--video', 'in.mp4', '--plan', 'plan.json', '--output', 'out.mp4',
                       '--template', str(self.path), '--rms-db', '-28', '--no-audio-enabled'])
        settings = render.call_args.kwargs['audio_settings']
        self.assertEqual(settings, {'max_cues': 1, 'rms_db': -28, 'enabled': False})
        plan = {'version': 1, 'cues': [{'id': 'cue', 'rms_db': -20}]}
        before = copy.deepcopy(plan)
        self.assertEqual(apply_audio_settings(plan, settings)['cues'], [])
        self.assertEqual(plan, before)
        self.assertEqual(apply_audio_settings(plan, {'duck_db': -6})['cues'][0]['rms_db'], -20)


if __name__ == '__main__':
    unittest.main()
