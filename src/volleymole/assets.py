"""Versioned presentation assets. Downloads are explicit and require only stdlib."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from urllib.request import Request, urlopen
import zipfile


MANIFEST = Path(__file__).with_name('asset_manifest.json')
BUNDLED_ROOT = Path(__file__).with_name('assets')


def load_manifest(path=None):
    data = json.loads(Path(path or MANIFEST).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not data.get('files'):
        raise ValueError('Invalid asset manifest')
    safe_name(data['version'])
    if '/' in data['version']:
        raise ValueError('Invalid asset version')
    for name, entry in data['files'].items():
        safe_name(name)
        validate_entry(entry)
    validate_entry(data['archive'])
    return data


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '\\' in name or
            any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe asset path: {name}')
    return path


def validate_entry(entry):
    if (type(entry.get('bytes')) is not int or entry['bytes'] < 0 or
            not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in entry['sha256'])):
        raise ValueError('Invalid asset size or SHA-256')


def cache_directory(manifest=None):
    data = manifest if manifest is not None else load_manifest()
    base = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache')).expanduser()
    return base/'volleymole'/'assets'/data['version']


def install_directory(manifest=None):
    override = os.environ.get('VOLLEYMOLE_ASSETS')
    return Path(override).expanduser() if override else cache_directory(manifest)


def asset_root():
    """Explicit override, installed release, then existing developer assets."""
    target = install_directory()
    if os.environ.get('VOLLEYMOLE_ASSETS') or target.is_dir():
        return target.resolve()
    if (BUNDLED_ROOT/'branding/volleymole.png').is_file():
        return BUNDLED_ROOT
    return target.resolve()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(directory=None, manifest=None):
    data = manifest if manifest is not None else load_manifest()
    root = Path(directory) if directory is not None else asset_root()
    errors = []
    for name, entry in data['files'].items():
        path = root/name
        if any(p.is_symlink() for p in (path, *path.parents)):
            errors.append(f'{name}: symlink is not a release asset')
        elif not path.is_file():
            errors.append(f'{name}: missing')
        elif path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            errors.append(f'{name}: size or SHA-256 mismatch')
    if errors:
        raise ValueError('素材校验失败；请运行 volleymole assets fetch，或使用新的安装目录：\n' + '\n'.join(errors))
    return {'version': data['version'], 'directory': str(root.resolve()),
            'files': len(data['files']), 'status': 'verified'}


def _copy_checked(stream, destination, entry):
    size, hashed = 0, hashlib.sha256()
    with Path(destination).open('wb') as output:
        while chunk := stream.read(1024*1024):
            size += len(chunk)
            if size > entry['bytes']:
                raise ValueError('Asset download exceeds its pinned size')
            hashed.update(chunk)
            output.write(chunk)
    if size != entry['bytes'] or hashed.hexdigest() != entry['sha256']:
        raise ValueError('Asset size or SHA-256 mismatch; installation unchanged')


def install(archive=None, directory=None, manifest=None):
    """Verify in a sibling staging directory, then publish by atomic rename.

    Existing valid installs are reused offline. Existing invalid directories are
    preserved for inspection; choose a new directory to recover from damage.
    """
    data = load_manifest(manifest)
    target = Path(directory) if directory is not None else install_directory(data)
    target = target.expanduser().absolute()
    if target.is_symlink():
        raise ValueError('Asset installation directory must not be a symlink')
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent/('.'+target.name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return {**verify(target, data), 'reused': True}
        with tempfile.TemporaryDirectory(prefix='.'+target.name+'.', dir=target.parent) as temporary:
            work = Path(temporary)
            packed = work/'bundle.zip'
            if archive is not None:
                with Path(archive).open('rb') as stream:
                    _copy_checked(stream, packed, data['archive'])
            else:
                url = data['archive']['url']
                if not url.startswith('https://'):
                    raise ValueError('Assets require an HTTPS download URL')
                print(f'Downloading {data["version"]} ({data["archive"]["bytes"]/1024**2:.1f} MiB)', flush=True)
                request = Request(url, headers={'User-Agent': 'VolleyMole-assets'})
                with urlopen(request, timeout=60) as stream:
                    if not stream.geturl().startswith('https://'):
                        raise ValueError('Asset download redirected away from HTTPS')
                    _copy_checked(stream, packed, data['archive'])
            staging = work/'contents'
            staging.mkdir()
            try:
                with zipfile.ZipFile(packed) as bundle:
                    members = bundle.infolist()
                    names = [entry.filename for entry in members]
                    if len(set(names)) != len(names) or set(names) != set(data['files']):
                        raise ValueError('Asset archive has missing, unexpected or duplicate members')
                    for info in members:
                        safe_name(info.filename)
                        mode = stat.S_IFMT(info.external_attr >> 16)
                        if info.is_dir() or mode not in (0, stat.S_IFREG):
                            raise ValueError('Only regular asset files can be installed')
                        expected = data['files'][info.filename]
                        if info.file_size != expected['bytes']:
                            raise ValueError('Asset member size mismatch')
                        out = staging/info.filename
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(info) as source:
                            _copy_checked(source, out, expected)
            except zipfile.BadZipFile as exc:
                raise ValueError('Invalid asset ZIP') from exc
            verify(staging, data)
            staging.rename(target)
    return {**verify(target, data), 'reused': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='安装与校验固定版本的插画、字体和转场素材；模型与录像单独管理')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('fetch', 'verify', 'import-file'):
        command = commands.add_parser(name)
        command.add_argument('--directory', type=Path, help='自定义素材根目录；运行时设置 VOLLEYMOLE_ASSETS 为同一路径')
        if name == 'import-file':
            command.add_argument('archive', type=Path, help='从 Release 手动下载的完整 ZIP')
    args = parser.parse_args(argv)
    if args.command == 'verify':
        result = verify(args.directory)
    else:
        result = install(getattr(args, 'archive', None), args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
