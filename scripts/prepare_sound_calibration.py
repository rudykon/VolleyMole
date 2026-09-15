#!/usr/bin/env python3
"""Retrieve original FSD50K dev/train waveforms by ZIP ranges, preserving labels.

Selection is fixed before inference: per target, the first 40 positive and 120
nominal negative numeric Freesound IDs, excluding every ESC-50 source ID.
No labels are created. ZIP CRC32 verifies bytes against the official directory.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib

BASE = 'https://zenodo.org/records/4060432/files/'
TARGETS = ('Laughter', 'Clapping', 'Cheering')
PARTS = [f'FSD50K.dev_audio.z{i:02}' for i in range(1, 6)] + ['FSD50K.dev_audio.zip']
TRUTH_MD5 = 'ca27382c195e37d2269c4c866dd73485'
METADATA_MD5 = 'b9ea0c829a411c1d42adb9da539ed237'
CHUNK = 512 * 1024
MIRROR_REVISION = 'ccf1acaff12f3f4a4c10052dddafbb6d22152c9b'


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def get_range(filename, start=None, length=1, retries=3):
    """Only accept the exact requested byte range; never download a full volume."""
    spec = f'bytes={start}-{start+length-1}' if start is not None else f'bytes=-{length}'
    request = urllib.request.Request(BASE+filename+'?download=1',
                                    headers={'Range': spec, 'User-Agent': 'VolleyMole-public-calibration/1'})
    for attempt in range(retries):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=90) as response:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if response.status != 206 or not match:
                    raise ValueError('Server did not honor range; refusing full-volume download')
                first, last, total = map(int, match.groups())
                expected_start = start if start is not None else total-length
                if first != expected_start or last-first+1 != length:
                    raise ValueError('Unexpected Content-Range')
                data = response.read(length+1)
                if len(data) != length:
                    raise ValueError('Truncated or oversized byte range')
                return data, total
        except (OSError, ValueError) as exc:
            if attempt == retries-1:
                raise
            delay = min(60, int(exc.headers.get('Retry-After', '5'))) if isinstance(exc, urllib.error.HTTPError) and exc.code == 429 else 2**attempt
            time.sleep(delay)


def fetch_file(filename, destination, expected_md5, workers=6):
    destination = Path(destination)
    if destination.exists() and digest(destination, 'md5') == expected_md5:
        return
    _, total = get_range(filename)
    jobs = [(offset, min(CHUNK, total-offset)) for offset in range(0, total, CHUNK)]
    blocks = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(get_range, filename, offset, size): offset for offset, size in jobs}
        for future in as_completed(pending):
            blocks[pending[future]] = future.result()[0]
    temporary = destination.with_suffix('.partial')
    with temporary.open('wb') as out:
        for offset, _ in jobs:
            out.write(blocks[offset])
    if digest(temporary, 'md5') != expected_md5:
        raise ValueError(f'Official metadata checksum mismatch: {filename}')
    temporary.replace(destination)


def parse_directory(data):
    entries, offset = {}, 0
    while offset < len(data):
        header = struct.unpack_from('<4s6H3I5H2I', data, offset)
        if header[0] != b'PK\x01\x02':
            raise ValueError('Invalid ZIP central directory')
        name_size, extra_size, comment_size = header[10:13]
        name = data[offset+46:offset+46+name_size].decode('utf-8')
        if name in entries:
            raise ValueError('Duplicate ZIP member')
        if name.startswith('FSD50K.dev_audio/') and name.endswith('.wav'):
            if not re.fullmatch(r'FSD50K\.dev_audio/\d+\.wav', name):
                raise ValueError('Unsafe member path')
            entries[name] = {'name': name, 'flags': header[3], 'compression': header[4],
                             'crc32': header[7], 'compressed_size': header[8],
                             'uncompressed_size': header[9], 'disk': header[13], 'offset': header[16]}
        offset += 46+name_size+extra_size+comment_size
    return entries


def select_cohorts(rows, excluded):
    available = sorted((r for r in rows if r['split'] == 'train' and r['fname'] not in excluded),
                       key=lambda r: int(r['fname']))
    cohorts = {}
    for target in TARGETS:
        positive = [r['fname'] for r in available if target in r['labels'].split(',')][:40]
        negative = [r['fname'] for r in available if target not in r['labels'].split(',')][:120]
        if len(positive) < 40 or len(negative) < 120:
            raise ValueError('Insufficient original dev/train labels for fixed protocol')
        cohorts[target] = {'positive_ids': positive, 'nominal_negative_ids': negative}
    return cohorts


def read_spanned(entry, sizes):
    disk, offset = entry['disk'], entry['offset']
    length = entry['compressed_size'] + 4096  # bounded headroom for the local header/extra fields
    pieces = []
    while length:
        size = min(length, sizes[disk]-offset)
        pieces.append(get_range(PARTS[disk], offset, size)[0])
        length -= size
        disk += 1
        offset = 0
    payload = b''.join(pieces)
    header = struct.unpack_from('<4s5H3I2H', payload)
    if header[0] != b'PK\x03\x04' or header[2] & 1:
        raise ValueError('Invalid or encrypted local ZIP member')
    name_length, extra_length = header[9:11]
    if payload[30:30+name_length].decode() != entry['name'] or header[3] != entry['compression']:
        raise ValueError('ZIP member differs from official directory')
    start = 30+name_length+extra_length
    packed = payload[start:start+entry['compressed_size']]
    if entry['compression'] == 8:
        raw = zlib.decompress(packed, -15)
    elif entry['compression'] == 0:
        raw = packed
    else:
        raise ValueError('Unsupported ZIP compression')
    if len(raw) != entry['uncompressed_size'] or zlib.crc32(raw) != entry['crc32']:
        raise ValueError('Official per-file size/CRC verification failed')
    return raw


def prepare(root, esc50, truth_zip, workers=6, plan_only=False, mirror=False):
    root = Path(root)
    previous_manifest = json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    previous_sources = {r['fname']: r.get('retrieved_from') for r in previous_manifest.get('clips', [])}
    downloads = root/'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    truth_zip = Path(truth_zip)
    truth_zip.parent.mkdir(parents=True, exist_ok=True)
    fetch_file('FSD50K.ground_truth.zip', truth_zip, TRUTH_MD5, workers)
    if digest(truth_zip, 'md5') != TRUTH_MD5:
        raise ValueError('Official ground-truth ZIP checksum mismatch')
    with zipfile.ZipFile(truth_zip) as archive:
        csv_bytes = archive.read('FSD50K.ground_truth/dev.csv')
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    if len(rows) != 40966:
        raise ValueError('Unexpected official dev row count')
    (root/'dev.csv').write_bytes(csv_bytes)
    esc_manifest = json.loads((Path(esc50)/'manifest.json').read_text())
    excluded = {r['src_file'] for r in esc_manifest['clips']}
    cohorts = select_cohorts(rows, excluded)
    ids = {item for cohort in cohorts.values() for items in cohort.values() for item in items}
    by_id = {r['fname']: r for r in rows}
    tail_path = downloads/'dev_tail.bin'
    if not tail_path.exists():
        tail_path.write_bytes(get_range(PARTS[-1], None, 65536)[0])
    tail = tail_path.read_bytes()
    end = tail.rfind(b'PK\x05\x06')
    eocd = struct.unpack_from('<4s4H2IH', tail, end)
    if eocd[1:3] != (5, 5) or eocd[4] != 40967:
        raise ValueError('Unexpected official multipart ZIP layout')
    directory_path = downloads/'dev_central_directory.bin'
    if not directory_path.exists():
        jobs = [(offset, min(CHUNK, eocd[5]-offset)) for offset in range(0, eocd[5], CHUNK)]
        blocks = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(get_range, PARTS[-1], eocd[6]+offset, size): offset for offset, size in jobs}
            for future in as_completed(pending):
                blocks[pending[future]] = future.result()[0]
        directory_path.write_bytes(b''.join(blocks[offset] for offset, _ in jobs))
    entries = parse_directory(directory_path.read_bytes())
    selected = [entries[f'FSD50K.dev_audio/{fname}.wav'] for fname in sorted(ids, key=int)]
    print(json.dumps({'selected_unique_clips': len(selected), 'compressed_audio_bytes': sum(e['compressed_size'] for e in selected),
                      'classes': {k: {n: len(v) for n, v in c.items()} for k, c in cohorts.items()}}, indent=2), flush=True)
    protocol = {'protocol': 'fsd50k-original-dev-train-id-order-40pos-120nominalneg-f2-v1',
                'source_url': 'https://zenodo.org/records/4060432', 'original_partition': 'dev/train',
                'original_ground_truth_sha256': digest(truth_zip), 'dev_csv_sha256': digest(root/'dev.csv'),
                'excluded_esc50_source_ids': sorted(excluded, key=int),
                'selection': 'Per target: first 40 positive and first 120 nominal negative numeric IDs after exclusions.',
                'threshold_selection': 'Maximize clip-level F2 on each fixed cohort; choose the highest threshold on a tie.',
                'new_annotations': False, 'network_training': False,
                'negative_label_limit': 'Target absent from original positive labels is nominal absence, not an exhaustive verified negative.',
                'cohorts': cohorts}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if plan_only:
        return protocol
    metadata_path = downloads/'FSD50K.metadata.zip'
    fetch_file('FSD50K.metadata.zip', metadata_path, METADATA_MD5, workers)
    with zipfile.ZipFile(metadata_path) as archive:
        metadata = json.loads(archive.read('FSD50K.metadata/dev_clips_info_FSD50K.json'))
    audio_dir = root/'audio'
    audio_dir.mkdir(exist_ok=True)
    sizes = {}
    mirror_ids = set()
    if mirror:
        missing = [entry for entry in selected
                   if not (audio_dir/Path(entry['name']).name).is_file()
                   or (audio_dir/Path(entry['name']).name).stat().st_size != entry['uncompressed_size']
                   or zlib.crc32((audio_dir/Path(entry['name']).name).read_bytes()) != entry['crc32']]
        mirror_ids = {Path(entry['name']).stem for entry in missing}
        if missing:
            environment = os.environ.copy()
            for key in ('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'https_proxy', 'http_proxy', 'all_proxy'):
                environment.pop(key, None)
            environment.update(HF_ENDPOINT='https://hf-mirror.com', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                               HF_HUB_DISABLE_XET='1', HF_HOME=str(downloads/'hf_cache'))
            print(f'Downloading the same {len(missing)} missing original WAVs from a pinned mirror; official CRC is mandatory.', flush=True)
            def download_one(entry):
                command = [str(Path(sys.executable).with_name('hf')), 'download', 'Fhrozen/FSD50k',
                           f"clips/dev/{Path(entry['name']).name}", '--repo-type', 'dataset',
                           '--revision', MIRROR_REVISION, '--local-dir', str(root/'mirror'), '--quiet']
                for attempt in range(3):
                    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
                    if result.returncode == 0:
                        return
                    if attempt == 2:
                        raise RuntimeError(f"{entry['name']}: mirror download failed: {result.stderr[-700:]}")
                    time.sleep(2**attempt)
            errors = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {pool.submit(download_one, entry): entry for entry in missing}
                for index, future in enumerate(as_completed(pending), 1):
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(str(exc))
                    if index % 20 == 0:
                        print(f'Mirror: {index}/{len(missing)} fixed original files; {len(errors)} errors', flush=True)
            if errors:
                raise RuntimeError('\n'.join(errors))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(get_range, filename): disk for disk, filename in enumerate(PARTS)}
            for future in as_completed(pending):
                sizes[pending[future]] = future.result()[1]
    result_rows, failures = [], []

    def retrieve(entry):
        fname = Path(entry['name']).stem
        path = audio_dir/f'{fname}.wav'
        if not path.exists() or path.stat().st_size != entry['uncompressed_size'] or zlib.crc32(path.read_bytes()) != entry['crc32']:
            raw = (root/'mirror'/'clips'/'dev'/f'{fname}.wav').read_bytes() if mirror else read_spanned(entry, sizes)
            if len(raw) != entry['uncompressed_size'] or zlib.crc32(raw) != entry['crc32']:
                raise ValueError('Mirror waveform differs from the official original ZIP member')
            temporary = path.with_suffix('.partial')
            temporary.write_bytes(raw)
            temporary.replace(path)
        return {**by_id[fname], 'audio_path': f'audio/{fname}.wav', 'sha256': digest(path),
                'original_metadata': metadata[fname], 'official_zip_member': entry,
                'retrieved_from': (f'https://hf-mirror.com/datasets/Fhrozen/FSD50k/resolve/{MIRROR_REVISION}/clips/dev/{fname}.wav'
                                   if fname in mirror_ids else previous_sources.get(fname) or BASE+PARTS[entry['disk']]+'?download=1'),
                'origin_volume': BASE+PARTS[entry['disk']]+'?download=1'}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(retrieve, entry): entry for entry in selected}
        for index, future in enumerate(as_completed(pending), 1):
            try:
                result_rows.append(future.result())
            except Exception as exc:
                failures.append({'member': pending[future]['name'], 'error': f'{type(exc).__name__}: {exc}'})
            if index % 10 == 0:
                print(f'{index}/{len(selected)} calibration clips retrieved; {len(failures)} failures', flush=True)
    manifest = {'schema_version': 1, 'dataset': 'FSD50K', 'protocol': protocol, 'complete': not failures,
                'expected_clips': len(selected), 'failures': failures,
                'metadata_zip_sha256': digest(metadata_path), 'central_directory_sha256': digest(directory_path),
                'retrieved_utc': previous_manifest.get('retrieved_utc', datetime.now(timezone.utc).isoformat()),
                'byte_verification': 'Original central-directory CRC32 and uncompressed byte count; observed waveform SHA-256.',
                'clips': sorted(result_rows, key=lambda r: int(r['fname']))}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    if failures:
        raise RuntimeError('Incomplete download; rerun to resume original fixed sample without substituting clips')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/public_benchmarks/audio/fsd50k_dev_calibration'))
    parser.add_argument('--esc50', type=Path, default=Path('data/public_benchmarks/audio/esc50'))
    parser.add_argument('--truth-zip', type=Path, default=Path('data/public_benchmarks/audio/fsd50k_metadata/FSD50K.ground_truth.zip'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--mirror', action='store_true', help='Fetch identical original WAVs using pinned hf-cli mirror and official ZIP CRC')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be in 1..8')
    prepare(args.root, args.esc50, args.truth_zip, args.workers, args.plan_only, args.mirror)


if __name__ == '__main__':
    main()
