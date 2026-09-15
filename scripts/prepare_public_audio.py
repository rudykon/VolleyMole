#!/usr/bin/env python3
"""Download pinned official ESC-50 audio and existing labels; never annotate clips."""
import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REVISION = '33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
URL = f'https://codeload.github.com/karolpiczak/ESC-50/zip/{REVISION}'
# Observed from the official pinned archive on 2026-09-12, not an upstream signature.
ARCHIVE_SHA256 = '661183a6f53ef04f12c9bd618fed0ddc1713280d6c94a5a5431e844ba6f6a21f'
DEFAULT_ROOT = Path('data/public_benchmarks/audio')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_rows(rows):
    """Require official balance and audit source overlap without changing labels."""
    if len(rows) != 2000 or len({r['filename'] for r in rows}) != 2000:
        raise ValueError('ESC-50 must contain 2,000 unique official clips')
    categories = Counter(r['category'] for r in rows)
    folds = Counter(int(r['fold']) for r in rows)
    if len(categories) != 50 or set(categories.values()) != {40}:
        raise ValueError('ESC-50 must contain 50 classes with 40 clips each')
    if folds != Counter({i: 400 for i in range(1, 6)}):
        raise ValueError('ESC-50 original five folds must be preserved')
    sources = defaultdict(set)
    for row in rows:
        if PurePosixPath(row['filename']).name != row['filename']:
            raise ValueError('Invalid official audio filename')
        sources[row['src_file']].add(int(row['fold']))
    return {source: sorted(folds) for source, folds in sources.items() if len(folds) > 1}


def prepare(root, archive=None):
    root = Path(root)
    archive = Path(archive) if archive else root/'downloads'/f'ESC-50-{REVISION}.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        request = urllib.request.Request(URL, headers={'User-Agent': 'VolleyMole-public-benchmark/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    archive_hash = sha256(archive)
    if ARCHIVE_SHA256 and archive_hash != ARCHIVE_SHA256:
        raise ValueError('Pinned ESC-50 archive checksum mismatch')
    dataset = root/'esc50'
    dataset.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        prefix = f'ESC-50-{REVISION}/'
        for item in bundle.infolist():
            if item.is_dir() or not item.filename.startswith(prefix):
                continue
            relative = PurePosixPath(item.filename[len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive path')
            keep = (len(relative.parts) == 2 and relative.parts[0] == 'audio' and relative.suffix == '.wav')
            keep = keep or str(relative) in {'LICENSE', 'README.md', 'meta/esc50.csv', 'meta/esc50-human.xlsx'}
            if keep:
                destination = dataset/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as stream, destination.open('wb') as out:
                    shutil.copyfileobj(stream, out)
    metadata = dataset/'meta/esc50.csv'
    with metadata.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    source_overlap = validate_rows(rows)
    manifest = {
        'schema_version': 1, 'dataset': 'ESC-50', 'revision': REVISION,
        'source_url': 'https://github.com/karolpiczak/ESC-50',
        'download_url': URL, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'archive_sha256': archive_hash, 'annotation_file': 'meta/esc50.csv',
        'annotation_sha256': sha256(metadata), 'license_file': 'LICENSE',
        'license_sha256': sha256(dataset/'LICENSE'),
        'license': 'CC BY-NC 3.0; ESC-10 subset CC BY; retain upstream per-clip attributions',
        'annotation_origin': 'Original dataset authors; no new human, assistant or pseudo labels',
        'annotation_scope': 'One category per 5-second clip; no exhaustive multilabel absence or event timestamps',
        'split_policy': 'All five original folds retained; no training or threshold tuning on these clips',
        'original_cross_fold_source_ids': source_overlap,
        'clips': [dict(row, fold=int(row['fold']), target=int(row['target']),
                       audio_path=f"audio/{row['filename']}", sha256=sha256(dataset/'audio'/row['filename']))
                  for row in rows],
    }
    (dataset/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    manifest = prepare(args.root, args.archive)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'clips'}, ensure_ascii=False, indent=2))
    print(f"Preserved {len(manifest['clips'])} original labeled clips")


if __name__ == '__main__':
    main()
