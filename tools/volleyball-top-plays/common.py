"""Small, explicit file and process utilities. No shell command interpolation."""
import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def functions_digest(path, names):
    """Avoid invalidating expensive previews for an unrelated render-only edit."""
    source=Path(path).read_text(encoding='utf-8')
    selected=[ast.dump(node,include_attributes=False) for node in ast.parse(source).body
              if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
    return hashlib.sha256('\n'.join(selected).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def run(command, log, cwd=None, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps([str(x) for x in command], ensure_ascii=False) + '\n')
        stream.flush()
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'子进程退出码 {result.returncode}，日志：{log}')


def probe(video):
    result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
                                                '-show_streams', '-of', 'json', str(video)]))
    v = next(s for s in result['streams'] if s['codec_type'] == 'video')
    rotation = next((int(s['rotation']) % 360 for s in v.get('side_data_list', []) if 'rotation' in s), 0)
    width, height = v['width'], v['height']
    if rotation in (90, 270):
        width, height = height, width
    return {'path': str(Path(video).resolve()), 'duration_sec': float(result['format']['duration']),
            'width': width, 'height': height, 'rotation': rotation,
            'frame_count': int(v.get('nb_frames', 0)), 'start_sec': float(result['format'].get('start_time', 0)),
            'video_start_sec': float(v.get('start_time', 0)), 'has_audio': any(s['codec_type'] == 'audio' for s in result['streams']),
            'nominal_fps': v['r_frame_rate'], 'average_fps': v['avg_frame_rate']}


class Stages:
    """Each successful stage is reusable only while inputs AND output hashes match."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory/'state.json'
        self.data = read_json(self.path) if self.path.exists() else {'version': 1, 'stages': {}}

    def execute(self, name, signature, callback):
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        old = self.data['stages'].get(name, {})
        if old.get('status') == 'complete' and old.get('signature') == key:
            if all(Path(p).is_file() and digest(p) == h for p, h in old.get('artifacts', {}).items()) and old.get('artifacts'):
                print(f'[{name}] 复用已验证结果', flush=True)
                return old['result']
        begin = time.monotonic()
        entry = {'status': 'running', 'signature': key, 'started_at': time.time()}
        self.data['stages'][name] = entry
        save_json(self.path, self.data)
        print(f'[{name}] 开始', flush=True)
        try:
            result, artifacts = callback()
            entry.update(status='complete', result=result,
                         artifacts={str(Path(p).resolve()): digest(p) for p in artifacts})
        except BaseException as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            entry['elapsed_sec'] = round(time.monotonic()-begin, 3)
            save_json(self.path, self.data)
        print(f'[{name}] 完成 ({entry["elapsed_sec"]:.1f}s)', flush=True)
        return result
