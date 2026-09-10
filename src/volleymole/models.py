"""Explicit model acquisition. No network access occurs during inference.

Only pinned bytes are accepted, including before loading pickle-based weights.
ZIP members are read individually; archive paths are never extracted to disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen
import zipfile

from .common import APP, ROOT, digest, read_json


class ModelRegistry:
    def __init__(self, directory=None, manifest=None):
        self.directory = Path(directory or os.environ.get('VOLLEYMOLE_MODELS', ROOT/'models')).resolve()
        self.manifest = read_json(manifest or APP/'model_manifest.json')
        self.entries = self.manifest['models']

    def target(self, name):
        relative = Path(self.entries[name]['path'])
        path = (self.directory/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.directory):
            raise ValueError('Model manifest contains an unsafe path')
        return path

    def verify(self, names=None):
        results = {}
        for name in names or self.entries:
            entry = self.entries[name]
            path = self.target(name)
            if not path.is_file():
                raise FileNotFoundError(f'Missing model {name}: {path}; run volleymole models fetch or import-file')
            if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                raise ValueError(f'Model hash mismatch: {name}; refusing to load unverified weights')
            results[name] = {'path': str(path), 'bytes': entry['bytes'], 'sha256': entry['sha256']}
        return results

    def install(self, name, stream):
        """Validate in a sibling temporary file, then atomically replace only this model."""
        target = self.target(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = self.entries[name]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+'.', suffix='.partial', delete=False) as out:
                temporary = Path(out.name)
                size, hashed = 0, hashlib.sha256()
                for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                    size += len(chunk)
                    if size > entry['bytes']:
                        raise ValueError(f'Oversized model download: {name}')
                    hashed.update(chunk)
                    out.write(chunk)
            if size != entry['bytes'] or hashed.hexdigest() != entry['sha256']:
                raise ValueError(f'Model hash mismatch: {name}; existing model unchanged')
            temporary.replace(target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return str(target)

    def fetch(self, names=None):
        groups = {}
        for name in names or self.entries:
            try:
                self.verify([name])
                print(f'{name}: verified, reused', flush=True)
                continue
            except (FileNotFoundError, ValueError):
                pass
            groups.setdefault(self.entries[name]['download'], []).append(name)
        for key, selected in groups.items():
            spec = self.manifest['downloads'][key]
            print(f'Downloading {key} for {", ".join(selected)}', flush=True)
            # Fixed public HTTPS sources only; credentials are neither needed nor logged.
            if not spec['url'].startswith('https://'):
                raise ValueError('Model downloads must use HTTPS')
            with tempfile.TemporaryFile() as temp:
                if shutil.which('curl'):
                    # curl handles this host's HTTPS proxy where urllib's TLS
                    # handshake fails, and enforces a total transfer deadline.
                    result = subprocess.run(['curl','--fail','--silent','--show-error','--location',
                        '--proto','=https','--proto-redir','=https','--connect-timeout','30',
                        '--retry','3','--retry-all-errors','--retry-delay','1','--retry-max-time','120',
                        '--max-time','900','--max-filesize',str(spec['max_bytes']),spec['url']],
                        stdout=temp, stderr=subprocess.PIPE, timeout=910)
                    if result.returncode:
                        raise RuntimeError(f'Model download failed: {key}, curl exit {result.returncode}')
                else:
                    with urlopen(spec['url'], timeout=90) as response:
                        count = 0
                        for chunk in iter(lambda: response.read(4*1024*1024), b''):
                            count += len(chunk)
                            if count > spec['max_bytes']:
                                raise ValueError(f'Download too large: {key}')
                            temp.write(chunk)
                if temp.tell() > spec['max_bytes']:
                    raise ValueError(f'Download too large: {key}')
                temp.seek(0)
                if spec.get('sha256'):
                    hashed = hashlib.file_digest(temp, 'sha256').hexdigest()
                    if hashed != spec['sha256']:
                        raise ValueError(f'Archive hash mismatch: {key}')
                    temp.seek(0)
                if spec['format'] == 'zip':
                    with zipfile.ZipFile(temp) as archive:
                        for name in selected:
                            entry = self.entries[name]
                            member = archive.getinfo(entry['member'])
                            if member.file_size != entry['bytes']:
                                raise ValueError(f'Archive member size mismatch: {name}')
                            with archive.open(member) as stream:
                                self.install(name, stream)
                else:
                    if len(selected) != 1:
                        raise ValueError('Raw download must contain exactly one model')
                    self.install(selected[0], temp)
        return self.verify(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('verify', 'fetch'):
        p = sub.add_parser(command)
        p.add_argument('names', nargs='*')
    p = sub.add_parser('import-file')
    p.add_argument('name')
    p.add_argument('source', type=Path)
    args = parser.parse_args(argv)
    registry = ModelRegistry(args.directory)
    if args.command == 'import-file':
        with args.source.open('rb') as stream:
            registry.install(args.name, stream)
        result = registry.verify([args.name])
    elif args.command == 'fetch':
        result = registry.fetch(args.names or None)
    else:
        result = registry.verify(args.names or None)
    print(json.dumps(result, indent=2))
