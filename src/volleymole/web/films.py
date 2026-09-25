"""Self-contained customer films. Reading the shelf never inspects run records."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import quote

SHELF = Path('outputs/published')
IDENTIFIER = re.compile(r'^[a-f0-9]{24}$')
LOCK = threading.RLock()


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=128)
def cached_hash(path, fingerprint):
    return sha256(Path(path))


def read(workspace, path):
    try:
        data = workspace.read_json(workspace.path(str(path), exist=True))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def folder(workspace, film_id):
    if not isinstance(film_id, str) or not IDENTIFIER.fullmatch(film_id):
        raise ValueError('无效的成片编号')
    path = workspace.root / SHELF / film_id
    if path.resolve() != path or any(p.is_symlink() for p in (workspace.root / 'outputs', path.parent, path)):
        raise ValueError('成片目录不能使用符号链接')
    return path


def detail(workspace, film_id):
    base = folder(workspace, film_id)
    for name in ('manifest.json', 'video.mp4', 'poster.jpg'):
        if (base / name).is_symlink() or not (base / name).is_file():
            raise ValueError('成片包不完整')
    data = read(workspace, base / 'manifest.json')
    try:
        video, poster = data['video'], data['poster']
        assert data['schema_version'] == 1 and data['id'] == film_id
        assert isinstance(data['title'], str) and data['title'].strip()
        assert math.isfinite(data['created_at']) and data['created_at'] > 0
        assert video['file'] == 'video.mp4' and poster['file'] == 'poster.jpg'
        assert video['codec'] == 'h264' and video['pixel_format'] == 'yuv420p'
        assert video['audio_codec'] in ('aac', None)
        assert math.isfinite(video['duration']) and video['duration'] > 0
        assert type(video['width']) is int and video['width'] > 0
        assert type(video['height']) is int and video['height'] > 0
        assert re.fullmatch(r'[a-f0-9]{64}', video['sha256'])
        stat = (base / 'video.mp4').stat()
        assert stat.st_size == video['bytes'] > 0
        fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        assert cached_hash(str(base / 'video.mp4'), fingerprint) == video['sha256']
        assert (base / 'poster.jpg').stat().st_size == poster['bytes'] > 0
        assert data['verification'] in ('passed', 'failed', 'unknown')
        assert isinstance(data['highlights'], list)
        assert all(isinstance(r, dict) and isinstance(r['title'], str)
                   and type(r['rank']) is int for r in data['highlights'])
        # Only customer-facing fields cross the API, even for hand-edited manifests.
        return {key: data[key] for key in ('schema_version', 'id', 'title', 'created_at',
                'match_date', 'collection', 'design_suite', 'verification')} | {
            'video': {key: video[key] for key in ('file', 'codec', 'pixel_format', 'audio_codec',
                       'duration', 'width', 'height', 'bytes', 'sha256')},
            'poster': {'file': 'poster.jpg', 'bytes': poster['bytes']},
            'highlights': [{'rank': r['rank'], 'title': r['title']} for r in data['highlights']],
        }
    except (AssertionError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('成片描述文件无效或视频已改变，请重新收录') from exc


def entries(workspace):
    base = workspace.root / SHELF
    if base.resolve() != base or not base.is_dir():
        return []
    rows = []
    for path in sorted(base.iterdir()):
        try:
            data = detail(workspace, path.name)
        except (ValueError, OSError):
            continue
        rows.append({'id': data['id'], 'film_id': data['id'], 'name': data['title'] + '.mp4',
                     'title': data['title'], 'path': str(SHELF / data['id'] / 'video.mp4'),
                     'size': data['video']['bytes'], 'modified': data['created_at'],
                     'category': 'outputs', 'kind': 'video'})
    return sorted(rows, key=lambda r: (-r['modified'], r['id']))


def media_info(workspace, value):
    path = workspace.path(value, exist=True)
    if path.parent.parent != workspace.root / SHELF or path.name != 'video.mp4':
        return None
    data = detail(workspace, path.parent.name)
    return {key: data['video'][key] for key in ('duration', 'width', 'height')} | {
        'status': 'ready', 'poster': '/media?path=' + quote(str(SHELF / data['id'] / 'poster.jpg'))}


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(path)], capture_output=True, text=True, timeout=30, check=True)
    data = json.loads(result.stdout)
    video = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})
    audio = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), {})
    duration = float(data.get('format', {}).get('duration') or video.get('duration') or 0)
    if (video.get('codec_name') != 'h264' or video.get('pix_fmt') != 'yuv420p'
            or audio.get('codec_name') not in ('aac', None) or not math.isfinite(duration) or duration <= 0):
        raise ValueError('成片须为 H.264 / yuv420p 视频和 AAC 音频，请先转换为网页兼容格式')
    width, height = int(video['width']), int(video['height'])
    rotation = next((s['rotation'] for s in video.get('side_data_list', []) if 'rotation' in s), 0)
    if abs(round(float(rotation))) % 180 == 90:
        width, height = height, width
    return {'codec': 'h264', 'pixel_format': 'yuv420p', 'audio_codec': audio.get('codec_name'),
            'duration': duration, 'width': width, 'height': height}


def verification(workspace, run, source, source_hash):
    delivery = read(workspace, run / 'delivery.json')
    if delivery.get('sha256') == source_hash and delivery.get('complete_decode') == 'passed':
        return 'passed'
    for name in ('verification_lively.json', 'verification.json'):
        report = read(workspace, run / name)
        for row in report.get('videos', []):
            if row.get('sha256') == source_hash:
                return 'passed' if report.get('status') == 'passed' and row.get('full_decode') == 'passed' else 'failed'
    return 'unknown'


def publish(workspace, source, *, title, run=None, inherited=None, cancelled=None):
    """Build a complete package in staging; atomically expose it after all checks."""
    source = workspace.path(str(source), exist=True)
    if source.suffix.lower() != '.mp4' or not source.is_file():
        raise ValueError('收录需要已完成的 MP4 成片')
    with LOCK:
        if cancelled and cancelled():
            raise ValueError('已取消成片收录')
        original = (source.stat().st_size, source.stat().st_mtime_ns)
        source_hash = sha256(source)
        film_id = source_hash[:24]
        destination = folder(workspace, film_id)
        if destination.exists():
            return detail(workspace, film_id)['id']
        info = probe(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.staging-', dir=destination.parent) as temporary:
            stage = Path(temporary)
            video = stage / 'video.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(source), '-map', '0:v:0',
                            '-map', '0:a:0?', '-c', 'copy', '-map_metadata', '-1', '-movflags', '+faststart',
                            str(video)], capture_output=True, timeout=600, check=True)
            exported = probe(video)
            if abs(exported['duration'] - info['duration']) > .15:
                raise ValueError('整理后的成片时长不一致')
            poster = stage / 'poster.jpg'
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-ss', str(min(info['duration']*.15, 8)),
                            '-i', str(video), '-frames:v', '1', '-vf', "scale='min(720,iw)':-2", '-threads', '1',
                            str(poster)], capture_output=True, timeout=60, check=True)
            if not poster.is_file() or not poster.read_bytes().startswith(b'\xff\xd8'):
                raise ValueError('无法生成真实视频封面')
            if (source.stat().st_size, source.stat().st_mtime_ns) != original or sha256(source) != source_hash:
                raise ValueError('源成片仍在改变，请处理完成后再收录')
            decision = read(workspace, run / 'edit_decision.json') if run else {}
            delivery = read(workspace, run / 'delivery.json') if run else {}
            config = read(workspace, run / 'run_config.json') if run else {}
            inherited = inherited or {}
            metadata = {'schema_version': 1, 'id': film_id, 'title': str(title or decision.get('title') or source.stem)[:120],
                        'created_at': time.time(), 'match_date': delivery.get('day') or config.get('date') or inherited.get('match_date'),
                        'collection': decision.get('collection', inherited.get('collection', 'highlights')),
                        'design_suite': delivery.get('design_suite') or config.get('design_suite') or inherited.get('design_suite'),
                        'verification': verification(workspace, run, source, source_hash) if run else 'unknown',
                        'video': exported | {'file': 'video.mp4', 'bytes': video.stat().st_size,
                                            'sha256': sha256(video)},
                        'poster': {'file': 'poster.jpg', 'bytes': poster.stat().st_size},
                        'highlights': [{'rank': int(r.get('rank', i+1)), 'title': str(r.get('title') or r.get('reason') or r.get('rally_id') or f'回合 {i+1}')[:300]}
                                       for i, r in enumerate(decision.get('selected', []))] or inherited.get('highlights', [])}
            (stage / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
            if cancelled and cancelled():
                raise ValueError('已取消成片收录')
            os.rename(stage, destination)
        detail(workspace, film_id)
        return film_id


def publish_run(workspace, value, *, title='', started=0, reused=False, cancelled=None):
    root = workspace.path(str(value), exist=True)
    if not root.is_dir() or root == workspace.root or root.is_relative_to(workspace.root / SHELF):
        raise ValueError('请选择后台剪辑任务目录')
    results = []
    excluded = {'archive', 'first_export_without_required_voice', 'frames', 'previews', 'segments', 'clips',
                'clips_lively', 'analytics', 'tracking', 'verification', 'verification_lively', 'evidence'}
    for directory, dirs, files in os.walk(root, followlinks=False):
        run = Path(directory)
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded and not (run/d).is_symlink()]
        report_name = next((name for name in ('delivery.json', 'render_report_lively.json', 'render_report.json') if name in files), None)
        if not report_name:
            continue
        # A final delivery is authoritative, including when its video is missing.
        report = read(workspace, run / report_name)
        try:
            source = workspace.path(report.get('output'), exist=True)
        except (ValueError, OSError):
            continue
        if not reused and (run/report_name).stat().st_mtime < started-1:
            continue
        results.append(publish(workspace, source, title=title, run=run, cancelled=cancelled))
        dirs[:] = []
    return list(dict.fromkeys(results))


def publish_job(root, job, log='', cancelled=None):
    """Only successful video-producing tasks may promote their own output."""
    if job.get('status') != 'succeeded':
        return []
    from .server import Workspace
    workspace = Workspace.__new__(Workspace)
    workspace.root = Path(root).resolve()
    values = job.get('values', {})
    if job.get('command') in {'run', 'match'}:
        if values.get('stop_after') in {'manifest', 'rank', 'replay'} or values.get('list'):
            return []
        title = job.get('title', '')
        if title in {'run', 'match', '单视频剪辑', '整场多局合辑'}:
            title = ''
        return publish_run(workspace, values.get('output'), title=title,
                           started=job.get('started_at') or job.get('created_at', 0),
                           reused=bool(re.search(r'\[render(?:_lively)?\] 复用已验证结果', log)), cancelled=cancelled)
    if job.get('command') == 'meme-audio':
        source = workspace.path(values.get('video'), exist=True)
        if source.parent.parent != workspace.root / SHELF or source.name != 'video.mp4':
            return []
        original = detail(workspace, source.parent.name)
        output = workspace.path(values.get('output'), exist=True)
        report = read(workspace, output.with_suffix('.audio.json'))
        if report.get('video_unchanged') is not True:
            return []
        if workspace.path(report.get('source')) != source or workspace.path(report.get('output')) != output:
            return []
        return [publish(workspace, output, title=original['title'] + ' · 配音版', inherited=original, cancelled=cancelled)]
    return []


def main():
    parser = argparse.ArgumentParser(description='将完成的剪辑收录到网页成片目录；不会删除后台原文件')
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--run', required=True)
    parser.add_argument('--title', default='')
    args = parser.parse_args()
    from .server import Workspace
    workspace = Workspace.__new__(Workspace)
    workspace.root = args.workspace.resolve()
    ids = publish_run(workspace, args.run, title=args.title)
    if not ids:
        raise SystemExit('未找到可收录的最终成片；请检查交付视频是否仍然存在。')
    print(json.dumps({'film_ids': ids}, ensure_ascii=False))


if __name__ == '__main__':
    main()
