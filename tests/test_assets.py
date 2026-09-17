"""Public asset installer contracts, using tiny archives and no real downloads."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from volleymole import assets


def entry(blob):
    return {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}


class Response(io.BytesIO):
    def geturl(self):
        return 'https://release-assets.githubusercontent.com/pinned.zip'


class AssetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root/'installed'
        self.manifest = self.root/'manifest.json'
        self.archive = self.root/'bundle.zip'
        self.files = {'fonts/OFL.txt': b'font license', 'illustrated/ball.png': b'fixture pixels'}
        self.data = {'schema_version': 1, 'version': 'assets-test',
                     'files': {name: entry(blob) for name, blob in self.files.items()}}
        self.pack()

    def pack(self, members=None):
        with warnings.catch_warnings(), zipfile.ZipFile(self.archive, 'w') as archive:
            warnings.simplefilter('ignore', UserWarning)
            for name, blob in members if members is not None else self.files.items():
                archive.writestr(name, blob)
        self.data['archive'] = {**entry(self.archive.read_bytes()), 'url': 'https://example.invalid/assets.zip'}
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.data))

    def install(self, archive=None):
        return assets.install(archive, self.target, self.manifest)

    def test_fetch_checks_all_files_then_reuses_install_offline(self):
        with patch.object(assets, 'urlopen', return_value=Response(self.archive.read_bytes())) as request:
            result = self.install()
        request.assert_called_once()
        self.assertEqual(result['files'], 2)
        self.assertFalse(result['reused'])
        for name, blob in self.files.items():
            self.assertEqual((self.target/name).read_bytes(), blob)
        with patch.object(assets, 'urlopen', side_effect=AssertionError('must reuse offline')):
            self.assertTrue(self.install()['reused'])
        self.assertEqual(assets.verify(self.target, self.data)['status'], 'verified')

    def test_offline_import_and_corruption_detection_preserve_existing_files(self):
        self.install(self.archive)
        path = self.target/'illustrated/ball.png'
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            assets.verify(self.target, self.data)
        with patch.object(assets, 'urlopen', side_effect=AssertionError('must not download')):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                self.install()
        self.assertEqual(path.read_bytes(), b'corrupt')

    def test_truncated_oversized_and_hash_mismatched_downloads_leave_no_install(self):
        payload = self.archive.read_bytes()
        for blob in (payload[:-1], payload+b'x', b'x'*len(payload)):
            with self.subTest(size=len(blob)), patch.object(assets, 'urlopen', return_value=Response(blob)):
                with self.assertRaises(ValueError):
                    self.install()
                self.assertFalse(self.target.exists())
                self.assertFalse([p for p in self.root.glob('.installed.*') if p.suffix != '.lock'])

    def test_interrupted_transfer_cleans_staging(self):
        class BrokenResponse(Response):
            def read(self, size):
                raise OSError('interrupted')
        with patch.object(assets, 'urlopen', return_value=BrokenResponse()):
            with self.assertRaises(OSError):
                self.install()
        self.assertFalse(self.target.exists())
        self.assertFalse([p for p in self.root.glob('.installed.*') if p.suffix != '.lock'])

    def test_member_hash_is_checked_even_with_valid_archive_hash(self):
        members = list(self.files.items())
        name, blob = members[-1]
        members[-1] = (name, b'x'*len(blob))
        self.pack(members)
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.install(self.archive)
        self.assertFalse(self.target.exists())

    def test_missing_extra_duplicate_and_traversal_members_are_rejected(self):
        normal = list(self.files.items())
        for members in (normal[:-1], normal+[('extra.txt', b'extra')], normal+[normal[0]],
                        normal+[('../escape.txt', b'escape')]):
            with self.subTest(members=[str(n) for n, _ in members]):
                self.pack(members)
                with self.assertRaisesRegex(ValueError, 'members'):
                    self.install(self.archive)
                self.assertFalse(self.target.exists())
                self.assertFalse((self.root/'escape.txt').exists())

    def test_archive_symlink_is_rejected_before_publication(self):
        info = zipfile.ZipInfo('illustrated/ball.png')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.pack([('fonts/OFL.txt', self.files['fonts/OFL.txt']), (info, self.files['illustrated/ball.png'])])
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.install(self.archive)
        self.assertFalse(self.target.exists())

    def test_manifest_paths_and_version_cannot_escape_install_root(self):
        for name in ('../escape', '/escape', 'fonts/../escape', 'fonts\\escape', 'fonts//escape'):
            with self.subTest(name=name):
                data = {**self.data, 'files': {name: entry(b'x')}}
                self.manifest.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    self.install(self.archive)
        self.data['version'] = '../escape'
        self.save()
        with self.assertRaises(ValueError):
            self.install(self.archive)
        self.assertFalse(self.target.exists())

    def test_existing_symlink_is_not_reused_or_overwritten(self):
        elsewhere = self.root/'elsewhere'; elsewhere.mkdir()
        self.target.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.install(self.archive)
        self.assertTrue(self.target.is_symlink())
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_verify_rejects_member_symlink_even_if_bytes_match(self):
        self.install(self.archive)
        path = self.target/'fonts/OFL.txt'
        original = self.root/'license.txt'; path.rename(original); path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            assets.verify(self.target, self.data)

    def test_http_url_is_rejected_without_contacting_network(self):
        self.data['archive']['url'] = 'http://example.invalid/zip'; self.save()
        with patch.object(assets, 'urlopen', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError, 'HTTPS'):
                self.install()

    def test_root_selection_prioritizes_override_then_cache_then_developer_files(self):
        bundled = self.root/'bundled'; (bundled/'branding').mkdir(parents=True)
        (bundled/'branding/volleymole.png').write_bytes(b'local')
        with patch.object(assets, 'MANIFEST', self.manifest), patch.object(assets, 'BUNDLED_ROOT', bundled), \
                patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.root/'cache'), 'VOLLEYMOLE_ASSETS': ''}):
            self.assertEqual(assets.asset_root(), bundled)
            cache = assets.cache_directory(); cache.mkdir(parents=True)
            self.assertEqual(assets.asset_root(), cache)
            with patch.dict(os.environ, {'VOLLEYMOLE_ASSETS': str(self.target)}):
                self.assertEqual(assets.asset_root(), self.target)

    def test_cli_help_does_not_require_inference_or_rendering_dependencies(self):
        source = Path(assets.__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-S', '-m', 'volleymole', 'assets', '--help'],
            env={**os.environ, 'PYTHONPATH': str(source)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('import-file', result.stdout)

    def test_pack_builder_is_repeatable_and_excludes_models(self):
        path = Path(__file__).resolve().parents[1]/'scripts/build_asset_bundle.py'
        spec = importlib.util.spec_from_file_location('build_asset_bundle_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        source = self.root/'source'; (source/'fonts').mkdir(parents=True)
        (source/'fonts/OFL.txt').write_bytes(b'license')
        with contextlib.redirect_stdout(io.StringIO()):
            for folder in ('a', 'b'):
                module.build(source, self.root/folder, 'assets-test', self.root/f'{folder}.json')
        self.assertEqual((self.root/'a/volleymole-assets-test.zip').read_bytes(),
                         (self.root/'b/volleymole-assets-test.zip').read_bytes())
        (source/'models').mkdir(); (source/'models/weights.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Not a presentation asset'):
            module.build(source, self.root/'bad', 'assets-test', self.root/'bad.json')


if __name__ == '__main__':
    unittest.main()
