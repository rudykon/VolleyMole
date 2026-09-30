"""Build a versioned local sumi snapshot; never replace a published package."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'sumi-v2.0.0'
sys.path.insert(0, str(ROOT / 'src'))


def fingerprint(path):
    with path.open('rb') as stream:
        hashed = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': hashed}


def build(output):
    from PIL import features
    from volleymole.assets import asset_root, verify
    from volleymole.templates import read_template
    assets = asset_root()
    verify(assets)  # Existing released artwork must still match its source.
    if not (assets / 'fonts/ukai.ttc').is_file():
        raise ValueError('请先部署已确认的完整 ukai.ttc 字体集合及许可证')
    for name in ('ARPHICPL.TXT', 'NOTICE-UKai.txt'):
        if not (assets / 'fonts' / name).is_file():
            raise ValueError(f'字体许可记录缺失：{name}')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / VERSION
    archive = output / f'{VERSION}.zip'
    checksum = output / f'{VERSION}.sha256'
    if any(path.exists() for path in (destination, archive, checksum)):
        raise FileExistsError('定稿包已存在；不覆盖原版，请使用新的输出目录或新版本号')
    with tempfile.TemporaryDirectory(prefix='.sumi-build-', dir=output) as temporary:
        stage = Path(temporary) / VERSION
        stage.mkdir()
        source = ROOT / 'src/volleymole'
        # Only the application tree is copied: no match data, credentials or runs.
        for path in sorted(source.rglob('*')):
            relative = path.relative_to(source)
            if 'assets' in relative.parts or '__pycache__' in relative.parts:
                continue
            if path.is_symlink():
                raise ValueError(f'源码不应含符号链接：{relative}')
            if not path.is_file() or path.suffix.lower() not in ('.py', '.json', '.md', '.txt', '.html', '.css', '.js', '.svg', '.png'):
                continue
            target = stage / 'src/volleymole' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for path in sorted(assets.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'素材不应含符号链接：{path.name}')
            if not path.is_file():
                continue
            if path.suffix.lower() not in ('.png', '.svg', '.ttf', '.otf', '.ttc', '.json', '.txt', '.md'):
                raise ValueError(f'非呈现素材：{path.name}')
            target = stage / 'src/volleymole/assets' / path.relative_to(assets)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for name in ('pyproject.toml', 'uv.lock', 'LICENSE', 'README.md'):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / 'templates/builtin', stage / 'templates/builtin')
        shutil.copyfile(ROOT / 'scripts/sumi_frozen.py', stage / 'sumi.py')
        shutil.copyfile(ROOT / 'docs/模板与配音.md', stage / '使用说明.md')
        shutil.copyfile(ROOT / 'src/volleymole/web/static/previews/sumi.png', stage / 'preview.png')
        shutil.copyfile(ROOT / 'templates/builtin/sumi.json', stage / 'template.json')
        read_template(stage / 'template.json')
        manifest = {
            'schema_version': 1, 'version': VERSION, 'approved_on': '2026-09-30',
            'label': '墨间回合 · 水墨模板定稿', 'language': 'zh',
            'font': {'path': 'src/volleymole/assets/fonts/ukai.ttc', 'face_index': 0,
                     'family': 'AR PL UKai CN', **fingerprint(assets / 'fonts/ukai.ttc')},
            'preview': {'item': {'rank': 1, 'title': '极低救球 · 接力救险'},
                        'top_k': 10, 'canvas': [720, 1280]},
            'render_dependencies': {name: importlib.metadata.version(name) for name in (
                'pillow', 'fonttools', 'numpy', 'opencv-python-headless')},
            'freetype_version': features.version_module('freetype2'),
            'python_version': sys.version.split()[0],
            'files': {p.relative_to(stage).as_posix(): fingerprint(p)
                      for p in sorted(stage.rglob('*')) if p.is_file()},
        }
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Smoke-test a real copy with no dependency on the workspace source tree.
        subprocess.run([sys.executable, str(stage / 'sumi.py'), '--preview',
                        str(Path(temporary) / 'reproduced.png')], cwd=temporary, check=True)
        packed = Path(temporary) / archive.name
        with zipfile.ZipFile(packed, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(stage.rglob('*')):
                if not path.is_file():
                    continue
                info = zipfile.ZipInfo(f'{VERSION}/{path.relative_to(stage).as_posix()}',
                                       date_time=(2026, 9, 30, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, path.read_bytes())
        # Verify the archive after extraction, including actual preview pixels.
        extraction = Path(temporary) / 'extracted'
        with zipfile.ZipFile(packed) as bundle:
            if bundle.testzip() is not None:
                raise ValueError('ZIP CRC 校验失败')
            bundle.extractall(extraction)
        subprocess.run([sys.executable, str(extraction / VERSION / 'sumi.py'), '--preview',
                        str(Path(temporary) / 'extracted-preview.png')], cwd=temporary, check=True)
        entry = fingerprint(packed)
        # All checks passed; publish the local directory, archive and checksum.
        shutil.move(stage, destination)
        shutil.move(packed, archive)
        checksum.write_text(f"{entry['sha256']}  {archive.name}\n", encoding='utf-8')
    print(json.dumps({'version': VERSION, 'archive': str(archive),
                      'MiB': round(entry['bytes'] / 1024**2, 2),
                      'files': len(manifest['files']), 'sha256': entry['sha256'],
                      'extracted_preview': 'pixel-identical'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'templates/bundles')
    build(parser.parse_args().output)
