"""Small manifest fixtures; real encoding is covered by publication integration tests."""
import hashlib
import json
from pathlib import Path
import time


def film_fixture(root, title='周末联赛 · 十佳球', verification='unknown'):
    film_id = 'a' * 24
    base = Path(root) / 'outputs/published' / film_id
    base.mkdir(parents=True, exist_ok=True)
    video = base / 'video.mp4'
    video.write_bytes(b'unit-test-video')
    poster = base / 'poster.jpg'
    poster.write_bytes(b'\xff\xd8unit-test-poster')
    document = {'schema_version': 1, 'id': film_id, 'title': title, 'created_at': time.time(),
                'match_date': '2026-09-25', 'collection': 'highlights', 'design_suite': 'atelier',
                'verification': verification, 'video': {'file': 'video.mp4', 'bytes': video.stat().st_size,
                'mtime_ns': video.stat().st_mtime_ns, 'sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
                'codec': 'h264', 'pixel_format': 'yuv420p', 'audio_codec': 'aac', 'duration': 10.,
                'width': 1080, 'height': 1920}, 'poster': {'file': 'poster.jpg', 'bytes': poster.stat().st_size},
                'highlights': [{'rank': 1, 'title': '精彩扣球'}]}
    (base / 'manifest.json').write_text(json.dumps(document, ensure_ascii=False))
    return film_id
