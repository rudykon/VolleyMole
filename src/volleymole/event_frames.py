"""Bounded disk spool fed by the existing shared inference decoder."""
import math
import time
from pathlib import Path
from .common import read_json, save_json


def tap(packets, directory, fps):
    import cv2
    from .common import digest
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    bucket, entries, next_time, published = None, [], 0., -1.
    try:
        for packet in packets:
            when = packet.time_sec
            group = int(when//10)
            if group != bucket:
                if bucket is not None: save_json(directory/f'index_{bucket:06d}.json', entries)
                bucket, entries = group, []
            if when+1e-6 >= next_time:
                pixels = packet.pixels
                pixels = cv2.resize(pixels, (768, max(2, round(pixels.shape[0]*768/pixels.shape[1]))))
                ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok: raise ValueError('共享视频证据编码失败')
                name = f'frame_{packet.index:09d}.jpg'
                path = directory/name
                temp = path.with_suffix('.tmp'); temp.write_bytes(encoded.tobytes()); temp.replace(path)
                entries.append({'time_sec': when, 'path': name, 'sha256': digest(path)})
                next_time = (math.floor(when*fps+1e-6)+1)/fps
            if when-published >= 1:
                save_json(directory/f'index_{bucket:06d}.json', entries)
                save_json(directory/'status.json', {'status': 'writing', 'through_sec': when})
                published = when
            yield packet
        if bucket is not None: save_json(directory/f'index_{bucket:06d}.json', entries)
        save_json(directory/'status.json', {'status': 'complete', 'through_sec': published})
    except BaseException:
        save_json(directory/'status.json', {'status': 'unavailable', 'through_sec': published})
        raise


def frames(directory, start, end, fps, width, deadline):
    import cv2
    from .common import digest
    directory = Path(directory)
    while True:
        if time.monotonic() >= deadline: raise TimeoutError('shared frame deadline')
        status = read_json(directory/'status.json')
        if status['status'] == 'unavailable': return None
        if status['status'] == 'complete' or status.get('through_sec', -1) >= end: break
        time.sleep(.1)
    entries = []
    for group in range(int(start//10), int(end//10)+1):
        path = directory/f'index_{group:06d}.json'
        if path.is_file(): entries.extend(read_json(path))
    result, next_time = [], start
    for entry in entries:
        when = entry['time_sec']
        if not next_time-1e-6 <= when < end: continue
        path = directory/entry['path']
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != entry['sha256']:
            raise ValueError('共享视频证据损坏')
        pixels = cv2.imread(str(path))
        if pixels is None: raise ValueError('共享视频证据无法解码')
        pixels = cv2.resize(pixels, (width, max(2, round(pixels.shape[0]*width/pixels.shape[1]))))
        result.append((when, pixels))
        next_time = start+(math.floor((when-start)*fps+1e-6)+1)/fps
    return result
