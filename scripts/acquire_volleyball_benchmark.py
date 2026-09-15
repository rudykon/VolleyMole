#!/usr/bin/env python3
"""Acquire author-supplied VNL-STES labels and complete frame sequences.

No labels are inferred or edited. Deterministic original-split subsets can be
encoded into silent videos using the original 25 FPS frames. The public archive
contains frames, not audio. HTTP byte ranges avoid downloading the entire 13 GB
archive. Only Python's standard library and ffmpeg/ffprobe are needed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import http.cookiejar
import io
import json
import math
from fractions import Fraction
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import urllib.request
import zipfile


PROJECT_URL = "https://hoangqnguyen.github.io/stes/"
ARCHIVE_URL = (
    "https://o365ust-my.sharepoint.com/:u:/g/personal/nqhoang_office_ust_ac_kr/"
    "EUNnXrlFpJlAnn9RdE_C7W8Bma49lgRbZ6mx7uXb09H93g?e=as0DBg&download=1"
)
PUBLIC_LINK = "https://bit.ly/vnlvolley1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class RangeReader(io.RawIOBase):
    """Seekable, bounded, buffered HTTP reader for zipfile (including ZIP64)."""

    def __init__(self, url: str, direct: bool, byte_limit: int):
        handlers = [urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())]
        if direct:
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)
        self.url = url
        self.pos = 0
        self.transferred = 0
        self.byte_limit = byte_limit
        self.cache_start = 0
        self.cache = b""
        data, headers = self._fetch(0, 0)
        self.size = int(headers["Content-Range"].split("/")[-1])
        self.etag = headers.get("ETag")
        self.last_modified = headers.get("Last-Modified")

    def _fetch(self, start: int, end: int):
        requested = end - start + 1
        if requested > 64 * 1024 * 1024:
            raise ValueError("Refusing an individual range larger than 64 MiB")
        if self.transferred + requested > self.byte_limit:
            raise ValueError("Download byte budget exhausted")
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "VolleyMole-public-benchmark/1"},
                )
                with self.opener.open(request, timeout=45) as response:
                    if response.status != 206:
                        raise ValueError("Server must support HTTP 206 byte ranges")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-{end}/"):
                        raise ValueError("Server returned unexpected byte range")
                    body = response.read(requested + 1)
                    if len(body) != requested:
                        raise ValueError("Truncated or oversized byte range")
                    # Preserve the public source URL in manifests, never redirected
                    # temporary download credentials or anonymous session cookies.
                    self.url = response.geturl()
                    self.transferred += len(body)
                    return body, response.headers
            except (OSError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError("Unreachable")

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("Negative offset")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.pos
        count = min(count, self.size - self.pos)
        if count <= 0:
            return b""
        cache_offset = self.pos - self.cache_start
        if not (0 <= cache_offset and cache_offset + count <= len(self.cache)):
            self.cache_start = self.pos
            end = min(self.size, self.pos + max(count, 4 * 1024 * 1024)) - 1
            self.cache, _ = self._fetch(self.pos, end)
            cache_offset = 0
        result = self.cache[cache_offset:cache_offset + count]
        self.pos += len(result)
        return result


def safe_target(root: Path, member: str) -> Path:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or "\\" in member:
        raise ValueError(f"Unsafe archive member: {member}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes destination")
    return target


def extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, root: Path) -> Path:
    target = safe_target(root, info.filename)
    if info.file_size > 128 * 1024 * 1024:
        raise ValueError("Unexpectedly large benchmark member")
    # zipfile validates each original member's CRC while decoding. Local files
    # can only be reused when they retain the original size and CRC.
    if target.is_file() and target.stat().st_size == info.file_size:
        import zlib
        if zlib.crc32(target.read_bytes()) & 0xFFFFFFFF == info.CRC:
            return target
    payload = archive.read(info)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target


def select_samples(splits: dict, split: str, count: int, per_match: int | None = None,
                   excluded: set[str] | None = None) -> list[dict]:
    """Choose IDs only; never use event labels, clip length, or model outputs."""
    if split not in ('train', 'val', 'test') or count < 0 or (per_match is not None and per_match < 1):
        raise ValueError('Invalid original-split sampling request')
    match_sets = {}
    for name, records in splits.items():
        ids = [row['video'] for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate original clip ID')
        match_sets[name] = {value.split('/')[0] for value in ids}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if match_sets[left] & match_sets[right]:
            raise ValueError('Original source match crosses partitions')
    rows = sorted((row for row in splits[split] if row['video'] not in (excluded or set())),
                  key=lambda row: row['video'])
    def spaced(group, number):
        if number > len(group):
            raise ValueError('Insufficient unused source IDs; refusing to substitute another partition')
        return [group[round(i*(len(group)-1)/max(1, number-1))] for i in range(number)]
    if per_match is None:
        return spaced(rows, count)
    chosen = []
    for match in sorted(match_sets[split]):
        chosen.extend(spaced([row for row in rows if row['video'].split('/')[0] == match], per_match))
    return chosen


def prepare_layout(output: Path, name: str) -> Path:
    """Independent manifests with the established raw/videos relative layout."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Prepared name must contain only letters, digits, underscores or hyphens')
    prepared = output/'prepared'/name
    prepared.mkdir(parents=True, exist_ok=True)
    for directory in ('raw', 'videos'):
        target = output/directory
        target.mkdir(parents=True, exist_ok=True)
        link = prepared/directory
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise ValueError('Existing prepared data link has a different target')
        else:
            link.symlink_to(Path('../..')/directory, target_is_directory=True)
    return prepared


def validate_encoded_video(probe: dict, frames: int, fps: float) -> None:
    streams = probe['streams']
    videos = [stream for stream in streams if stream['codec_type'] == 'video']
    if len(videos) != 1 or any(stream['codec_type'] == 'audio' for stream in streams):
        raise ValueError('Derivative must contain exactly one video stream and no invented audio')
    stream = videos[0]
    if int(stream['nb_frames']) != frames:
        raise ValueError('Encoding duplicated or dropped original frames')
    if any(not math.isclose(float(Fraction(stream[key])), fps, abs_tol=1e-9)
           for key in ('r_frame_rate', 'avg_frame_rate')):
        raise ValueError('Encoding changed the original frame rate')
    if (not math.isclose(float(stream['duration']), frames/fps, abs_tol=.001)
            or abs(float(stream.get('start_time', 0))) > 1e-6):
        raise ValueError('Encoding changed the original time origin or duration')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/public_benchmarks/volleyball/vnl_stes"))
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument("--sample-count", type=int, default=5, help="Deterministically spaced source IDs; 0 for labels only")
    parser.add_argument('--samples-per-match', type=int, help='Choose this many spaced IDs per original match, overriding sample-count')
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[], help='Exclude all previously observed sample IDs from this manifest')
    parser.add_argument('--prepared-name', help='Independent prepared/<name>/manifest.json; never overwrites the legacy root manifest')
    parser.add_argument('--sealed', action='store_true', help='Mark an untouched final test subset; never print its events')
    parser.add_argument("--direct", action="store_true", help="Ignore environment HTTP proxies")
    parser.add_argument("--max-download-mib", type=int, default=2048)
    parser.add_argument("--no-video", action="store_true", help="Keep source JPEG frames without encoding silent video")
    args = parser.parse_args()
    if args.sample_count < 0 or args.max_download_mib <= 0 or (args.samples_per_match is not None and args.samples_per_match < 1):
        parser.error("sample-count must be nonnegative and max-download-mib positive")
    if args.sealed and args.split != 'test': parser.error('Only the original test split can be sealed')

    args.output.mkdir(parents=True, exist_ok=True)
    name = args.prepared_name or (f'{args.split}_per_match_{args.samples_per_match}'
                                  if args.samples_per_match else f'{args.split}_{args.sample_count}')
    prepared = prepare_layout(args.output, name)
    prior = args.output/'manifest.json'
    prior_hash = sha256(prior) if prior.is_file() else None
    exclusions, exclusion_sources = set(), []
    for path in args.exclude_manifest:
        excluded_manifest = json.loads(path.read_text())
        if excluded_manifest.get('dataset') != 'VNL-STES': raise ValueError('Exclusion manifest is not VNL-STES')
        exclusions.update(row['video'] for row in excluded_manifest['samples'])
        exclusion_sources.append({'path': str(path), 'sha256': sha256(path), 'samples': len(excluded_manifest['samples'])})
    reader = RangeReader(ARCHIVE_URL, args.direct, args.max_download_mib * 1024 * 1024)
    print(f"Reading public VNL archive directory ({reader.size:,} archive bytes)", flush=True)
    archive = zipfile.ZipFile(reader)
    infos = archive.infolist()
    inventory = [{'member': item.filename, 'bytes': item.file_size, 'compressed_bytes': item.compress_size,
                  'zip_crc32': f'{item.CRC:08x}', 'header_offset': item.header_offset} for item in infos]
    inventory_path = args.output/'archive_inventory.json'
    write_json(inventory_path, inventory)
    labels = [i for i in infos if i.filename.endswith((".json", ".txt", ".csv"))]
    frames = [i for i in infos if i.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
    media = [i.filename for i in infos if i.filename.lower().endswith((".mp4", ".wav", ".mp3", ".mkv", ".webm"))]
    artifacts = []
    for info in sorted(labels, key=lambda i: i.header_offset):
        path = extract_member(archive, info, args.output / "raw")
        artifacts.append({"path": str(path.relative_to(args.output)), "archive_member": info.filename,
                          "bytes": path.stat().st_size, "sha256": sha256(path), "zip_crc32": f"{info.CRC:08x}"})
    print(f"Preserved {len(labels)} original label files", flush=True)

    test_candidates = [i.filename for i in labels if PurePosixPath(i.filename).name == "test.json"]
    if len(test_candidates) != 1:
        raise ValueError("Expected one unambiguous original test split")
    annotation_root = PurePosixPath(test_candidates[0]).parent
    raw_root = args.output / "raw" / str(annotation_root)
    splits = {s: json.loads((raw_root / f"{s}.json").read_text()) for s in ("train", "val", "test")}
    groups = {s: sorted({r["video"].split("/")[0] for r in records}) for s, records in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(groups[left]) & set(groups[right]):
            raise ValueError(f"Original match leakage between {left} and {right}")
    summary = {}
    for split, records in splits.items():
        counts = collections.Counter(e["label"] for r in records for e in r["events"])
        for row in records:
            if not math.isfinite(row["fps"]) or row["fps"] <= 0:
                raise ValueError("Invalid source FPS")
            if any(not 0 <= event["frame"] < row["num_frames"] for event in row["events"]):
                raise ValueError("Source event is outside its frame sequence")
        summary[split] = {"clips": len(records), "events": sum(counts.values()),
                          "frames": sum(r["num_frames"] for r in records), "classes": dict(counts), "matches": groups[split],
                          "stale_num_events_fields": [{"video": r["video"], "num_events": r["num_events"],
                                                       "actual_events": len(r["events"])} for r in records
                                                      if r["num_events"] != len(r["events"])]}

    chosen = select_samples(splits, args.split, args.sample_count, args.samples_per_match, exclusions)
    previous_prepared = prepared/'manifest.json'
    if previous_prepared.is_file():
        saved = json.loads(previous_prepared.read_text())
        if saved['sample_selection']['split'] != args.split or [r['video'] for r in saved['samples']] != [r['video'] for r in chosen]:
            raise ValueError('Refusing to replace a prepared manifest with a different frozen selection')
    prior_selection = prepared/'selection.json'
    if prior_selection.is_file():
        frozen = json.loads(prior_selection.read_text())
        if frozen['split'] != args.split or frozen['selected_video_ids'] != [row['video'] for row in chosen]:
            raise ValueError('Refusing to change fixed sample IDs after an interrupted acquisition')
    write_json(prepared/'selection.json', {'dataset': 'VNL-STES', 'split': args.split,
        'selected_video_ids': [row['video'] for row in chosen], 'sealed': args.sealed,
        'method': 'sort_video_id_then_evenly_spaced_indices_per_match' if args.samples_per_match else 'sort_video_id_then_evenly_spaced_indices',
        'uses_event_labels': False, 'excluded_manifests': exclusion_sources})
    selected = []
    for row in chosen:
        video = row["video"]
        matching = [i for i in frames if f"/{video}/" in i.filename]
        if not matching:
            raise ValueError(f"Missing original frame sequence for {video}")
        frame_paths = []
        frame_hash = hashlib.sha256()
        for info in sorted(matching, key=lambda i: i.header_offset):
            path = extract_member(archive, info, args.output / "raw")
            frame_paths.append(path)
        frame_paths.sort(key=lambda p: int(p.stem))
        numbers = [int(p.stem) for p in frame_paths]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError(f"Nonconsecutive frames for {video}")
        if numbers[0] != 0:
            raise ValueError('Source frames must start at zero; refusing to shift original annotation times')
        for path in frame_paths:
            frame_hash.update(f"{path.name} {sha256(path)}\n".encode())
        if row['fps'] != 25: raise ValueError('This source release is expected to retain its original 25 FPS')
        if any(not 0 <= event['frame'] < len(frame_paths) for event in row['events']):
            raise ValueError('Original action lies outside the complete available sequence')
        record = {"video": video, "match_id": video.split("/")[0], "split": args.split,
                  "original_annotation": str((raw_root / f'{args.split}.json').relative_to(args.output)),
                  'original_annotation_sha256': sha256(raw_root/f'{args.split}.json'),
                  'annotation_row_sha256': hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                  "frame_directory": str(frame_paths[0].parent.relative_to(args.output)),
                  "frames": len(frame_paths), "fps": row["fps"], "first_frame_file": frame_paths[0].name,
                  "source_declared_num_frames": row["num_frames"],
                  "frame_count_matches_annotation": len(frame_paths) == row["num_frames"],
                  "source_frame_hash_manifest_sha256": frame_hash.hexdigest(),
                  "audio_available": False, "source_full_match_offset_seconds": None,
                  "label_origin": "author_model_proposals_human_corrected", "annotation_row": row}
        if not args.no_video:
            output_video = args.output / "videos" / f"{video.replace('/', '__')}.mp4"
            output_video.parent.mkdir(parents=True, exist_ok=True)
            pattern = frame_paths[0].parent / (f"%0{len(frame_paths[0].stem)}d{frame_paths[0].suffix}")
            temp_video = output_video.with_suffix('.tmp.mp4')
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(row["fps"]),
                            "-start_number", str(numbers[0]), "-i", str(pattern), "-frames:v", str(len(frame_paths)),
                            "-fps_mode", 'passthrough', '-threads', '2',
                            "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(temp_video)], check=True)
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(temp_video)]))
            validate_encoded_video(probe, len(frame_paths), row['fps'])
            temp_video.replace(output_video)
            probe['format']['filename'] = str(output_video)
            record.update({"derived_video": str(output_video.relative_to(args.output)),
                           "derived_video_sha256": sha256(output_video), "ffprobe": probe,
                           "video_derivation": "lossy H264 encoding of complete original JPEG sequence at source fps, without audio"})
        selected.append(record)
        print(f"Acquired complete {args.split} sequence {len(selected)}/{len(chosen)}: {len(frame_paths)} frames", flush=True)

    manifest = {"schema_version": 1, "dataset": "VNL-STES", "project_url": PROJECT_URL,
                "public_download_link": PUBLIC_LINK, "public_archive_url": ARCHIVE_URL,
                "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "archive_bytes": reader.size, "archive_etag": reader.etag, "archive_last_modified": reader.last_modified,
                "archive_root": str(annotation_root), "archive_media_members": media,
                "downloaded_bytes_this_run": reader.transferred,
                'archive_inventory': {'path': str(inventory_path), 'sha256': sha256(inventory_path), 'members': len(inventory)},
                'legacy_manifest_sha256_preserved': prior_hash,
                "original_annotation_files": artifacts, "splits": summary, "samples": selected,
                "sample_selection": {"split": args.split, "method": "sort_by_video_id_then_evenly_spaced_indices",
                                     "requested_count": len(chosen) if args.samples_per_match else args.sample_count, "uses_event_labels": False,
                                     'per_original_match': args.samples_per_match, 'excluded_manifests': exclusion_sources,
                                     'sealed_for_final_evaluation': args.sealed,
                                     "is_full_test_set": args.split == 'test' and len(selected) == len(splits['test'])},
                "license": {"data_license": "No standalone data license included in the public archive; preserve author copyright and attribution",
                            "source_permission_statement": "Authors state matches were sourced with permission from Volleyball World",
                            "code_license_is_not_data_license": True},
                "limitations": ["Source archive has no audio or original full-match time offsets",
                                "Action labels are not highlight ratings or blooper labels",
                                "Rallies were extracted by the authors' court visibility algorithm, so clip edges are not independently human-labelled complete rally boundaries",
                                "Public ZIP is named vnl_1.0.zip but contains vnl_1.5; actual training event counts differ from the publication",
                                "Some source frame counts or num_events summaries are stale; original data is preserved and discrepancies are recorded",
                                "Current test split contains one match only; broader match generalization needs additional source matches"]}
    write_json(prepared/'manifest.json', manifest)
    if prior_hash is not None and sha256(prior) != prior_hash:
        raise ValueError('Legacy observed-test manifest unexpectedly changed')
    print(json.dumps({"manifest": str(prepared/'manifest.json'), 'split': args.split,
                      "sample_count": len(selected), "downloaded_bytes": reader.transferred}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
