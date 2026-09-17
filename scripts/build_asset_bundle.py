"""Build the complete presentation-only Release ZIP and pinned manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volleymole.assets import safe_name


def build(source, output, version, manifest):
    safe_name(version)
    if '/' in version:
        raise ValueError('Version must be a single path component')
    files = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in asset source: {path}')
        if not path.is_file():
            continue
        name = path.relative_to(source).as_posix()
        if (name != 'README.md' and name.split('/')[0] not in
                ('fonts', 'branding', 'illustrated', 'transitions', 'design_suites')):
            raise ValueError(f'Not a presentation asset: {name}')
        if path.suffix.lower() not in ('.png', '.svg', '.ttf', '.otf', '.json', '.txt', '.md'):
            raise ValueError(f'Unexpected asset type: {name}')
        blob = path.read_bytes()
        files[name] = {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
    if not files:
        raise ValueError('Empty asset source')
    output.mkdir(parents=True, exist_ok=True)
    name = f'volleymole-{version}.zip'
    packed = output/name
    if packed.exists():
        raise FileExistsError('Use a new output directory; published assets must not be replaced')
    with zipfile.ZipFile(packed, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative in files:
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            blob = (source/relative).read_bytes()
            if hashlib.sha256(blob).hexdigest() != files[relative]['sha256']:
                raise ValueError(f'Asset changed during packaging: {relative}')
            bundle.writestr(info, blob)
    hashed = hashlib.sha256(packed.read_bytes()).hexdigest()
    data = {'schema_version': 1, 'version': version,
            'archive': {'url': f'https://github.com/rudykon/VolleyMole/releases/download/{version}/{name}',
                        'bytes': packed.stat().st_size, 'sha256': hashed}, 'files': files}
    encoded = json.dumps(data, ensure_ascii=False, indent=2)+'\n'
    manifest.write_text(encoded, encoding='utf-8')
    (output/'asset_manifest.json').write_text(encoded, encoding='utf-8')
    manifest_hash = hashlib.sha256(encoded.encode()).hexdigest()
    (output/'SHA256SUMS').write_text(f'{hashed}  {name}\n{manifest_hash}  asset_manifest.json\n')
    print(json.dumps({'archive': str(packed), 'files': len(files), 'MiB': round(packed.stat().st_size/1024**2, 2),
                      'sha256': hashed}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'src/volleymole/assets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', default='assets-v1')
    parser.add_argument('--manifest', type=Path, default=ROOT/'src/volleymole/asset_manifest.json')
    args = parser.parse_args()
    build(args.source, args.output, args.version, args.manifest)
