"""Media roles follow purpose and provenance, not extensions or task titles."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from volleymole.web.server import Workspace


class LibraryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.app = Workspace.__new__(Workspace)
        self.app.root = self.root

    def write(self, name, value=b'video'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)) if isinstance(value, (dict, list)) else path.write_bytes(value)
        return path

    def paths(self):
        return {kind: {r['path'] for r in rows} for kind, rows in self.app.library().items()}

    def test_same_extension_has_three_distinct_roles_and_orphans_are_not_films(self):
        for path in ('data/2026.9.15.1.MOV', 'data/meme-assets/raw/nice.mp4',
                     'data/uploads/materials/batch/short.mp4', 'data/uploads/sources/batch/match.mp4',
                     'data/uploads/batch/voice.wav', 'outputs/orphan.mp4', 'runs/game/final.mp4',
                     'outputs/segments/part.mp4'):
            self.write(path)
        self.write('runs/game/render_report.json', {'output': 'runs/game/final.mp4'})
        paths = self.paths()
        self.assertEqual(paths['outputs'], set())
        self.assertEqual(paths['sources'], {'data/2026.9.15.1.MOV', 'data/uploads/sources/batch/match.mp4'})
        self.assertEqual(paths['materials'], {'data/meme-assets/raw/nice.mp4',
                                             'data/uploads/materials/batch/short.mp4', 'data/uploads/batch/voice.wav'})
        self.assertTrue(all(not paths[a] & paths[b] for a, b in
                            [('outputs', 'sources'), ('outputs', 'materials'), ('sources', 'materials')]))

    def test_successful_mixing_of_source_or_asset_cannot_become_a_finished_film(self):
        for source in ('data/2026.9.15.1.mp4', 'data/meme-assets/raw/nice.mp4'):
            self.write(source)
            self.write('outputs/mixed.mp4')
            self.write('outputs/mixed.audio.json', {'source': source, 'output': 'outputs/mixed.mp4', 'video_unchanged': True})
            self.app.jobs = SimpleNamespace(list=lambda: [{'command': 'meme-audio', 'status': 'succeeded',
                'title': '本地配音混音', 'values': {'video': source, 'output': 'outputs/mixed.mp4'}}])
            self.assertEqual(self.paths()['outputs'], set())

    def test_legacy_audio_exports_do_not_publish_themselves(self):
        self.write('runs/game/final.mp4')
        self.write('runs/game/render_report.json', {'output': 'runs/game/final.mp4'})
        for path, source in [('outputs/voiced.mp4', 'runs/game/final.mp4'),
                             ('outputs/revised.mp4', 'outputs/voiced.mp4'),
                             ('runs/preview/mix.mp4', 'runs/game/final.mp4')]:
            self.write(path)
            self.write(str(Path(path).with_suffix('.audio.json')), {'source': source, 'output': path, 'video_unchanged': True})
        self.assertEqual(self.paths()['outputs'], set())

    def test_legacy_delivery_reports_cannot_populate_the_shelf(self):
        for path in ('outputs/clean.mp4', 'outputs/final.mp4', 'runs/game/archive/final.mp4'):
            self.write(path)
        self.write('runs/game/render_report.json', {'output': 'outputs/clean.mp4'})
        self.write('runs/game/delivery.json', {'output': 'outputs/final.mp4'})
        self.write('runs/game/archive/delivery.json', {'output': 'outputs/final.mp4'})
        self.assertEqual(self.paths()['outputs'], set())

    def test_valid_recorded_export_never_adds_a_same_name_backup(self):
        self.write('outputs/final.mp4')
        self.write('runs/game/final.mp4')
        self.write('runs/game/render_report.json', {'output': 'outputs/final.mp4'})
        self.assertEqual(self.paths()['outputs'], set())

    def test_missing_delivery_does_not_promote_clean_intermediate(self):
        self.write('outputs/clean.mp4')
        self.write('runs/game/render_report.json', {'output': 'outputs/clean.mp4'})
        self.write('runs/game/delivery.json', {'output': 'outputs/missing.mp4'})
        self.assertEqual(self.paths()['outputs'], set())

    def test_asset_catalog_disambiguates_video_assets_and_rejects_private_paths(self):
        self.write('data/short.mp4')
        self.write('data/secret.mp4')
        self.write('data/catalog.json', {'assets': [{'path': 'data/short.mp4'}, {'path': 'data/secret.mp4'}, None]})
        self.write('data/invalid/catalog.json', {'assets': None})
        paths = self.paths()
        self.assertEqual(paths['materials'], {'data/short.mp4'})
        self.assertEqual(paths['sources'], set())

    def test_files_under_data_keep_their_explicit_role_until_published(self):
        self.write('data/uploads/materials/batch/final.mp4')
        self.write('runs/game/render_report.json', {'output': 'data/uploads/materials/batch/final.mp4'})
        paths = self.paths()
        self.assertEqual(paths['outputs'], set())
        self.assertEqual(paths['materials'], {'data/uploads/materials/batch/final.mp4'})
