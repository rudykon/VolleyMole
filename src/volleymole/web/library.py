"""Customer films use published packages; raw recordings and assets stay separate."""
import os
from pathlib import Path

from .films import entries
from .media import AUDIO, VIDEO

INTERNAL = {'analytics', 'tracking', 'previews', 'profiles', 'frames', 'segments',
            'clips', 'clips_lively', 'semantic_cache', 'media_cache', 'verification',
            'verification_lively', 'ocr_runtime', 'visual-assets', 'web-config',
            'public_benchmarks', '__pycache__'}


def collections(workspace):
    rows = {kind: {} for kind in ('sources', 'materials')}

    def add(value, kind):
        try:
            path = workspace.path(str(value), exist=True)
            relative = path.relative_to(workspace.root)
            if (not relative.is_relative_to('data') or INTERNAL.intersection(relative.parts)
                    or not path.is_file() or path.suffix.lower() not in VIDEO | AUDIO
                    or not path.stat().st_size):
                return
            if path.suffix.lower() in AUDIO:
                kind = 'materials'
            key = workspace.relative(path)
            stat = path.stat()
            rows[kind][key] = {'name': path.name, 'path': key, 'size': stat.st_size,
                               'modified': stat.st_mtime, 'category': kind,
                               'kind': 'video' if path.suffix.lower() in VIDEO else 'audio'}
        except (ValueError, OSError):
            pass

    start = workspace.root / 'data'
    if not start.is_symlink():
        for directory, dirs, files in os.walk(start, followlinks=False):
            folder = Path(directory)
            try:
                workspace.path(str(folder), exist=True)
            except (ValueError, OSError):
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in INTERNAL
                             and not (folder / d).is_symlink())
            relative = folder.relative_to(workspace.root)
            asset = any(relative.is_relative_to(p) for p in
                        ('data/meme-assets', 'data/materials', 'data/uploads/materials'))
            for name in sorted(files):
                path = folder / name
                if path.suffix.lower() in VIDEO | AUDIO:
                    add(path, 'materials' if asset else 'sources')
                elif name == 'catalog.json':
                    try:
                        document = workspace.read_json(workspace.path(str(path), exist=True))
                        assets = document.get('assets', []) if isinstance(document, dict) else []
                        for item in assets if isinstance(assets, list) else []:
                            if isinstance(item, dict) and isinstance(item.get('path'), str):
                                candidate = workspace.root / item['path']
                                add(candidate if candidate.is_file() else folder / item['path'], 'materials')
                    except (ValueError, OSError):
                        pass
    for key in rows['materials']:
        rows['sources'].pop(key, None)
    return {'outputs': entries(workspace)} | {
        kind: sorted(items.values(), key=lambda r: (-r['modified'], r['name'], r['path']))
        for kind, items in rows.items()}
