"""Local sound measurements; loudness and transient hypotheses are not labels."""
from pathlib import Path
import subprocess
import hashlib
import json
import fcntl
import time
import numpy as np


def cached_audio(source, cache, deadline, fresh=False):
    """Share decoded PCM and match-background measurements across top-k outputs."""
    from .common import APP, digest, read_json, save_json
    key = hashlib.sha256(json.dumps({'source': source['identity'], 'code': digest(APP/'audio_events.py')}, sort_keys=True).encode()).hexdigest()
    directory = Path(cache)/'audio'/key; directory.mkdir(parents=True, exist_ok=True)
    path = directory/'waveform.npy'; summary = directory/'features.json'
    with (directory/'.lock').open('a') as lock:
        while True:
            try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() >= deadline: raise TimeoutError('audio cache deadline')
                time.sleep(.1)
        if not fresh and summary.is_file():
            saved = read_json(summary)
            if saved['waveform_sha256'] is None: return None, saved['features']
            if path.is_file() and digest(path) == saved['waveform_sha256']:
                return np.load(path, mmap_mode='r', allow_pickle=False), saved['features']
        remaining = deadline-time.monotonic()
        if remaining <= 0: raise TimeoutError('audio deadline')
        audio = decode_audio(source, min(120, remaining))
        features = measure_audio(audio)
        if audio is not None:
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as stream: np.save(stream, audio, allow_pickle=False)
            temp.replace(path)
        save_json(summary, {'features': features, 'waveform_sha256': digest(path) if audio is not None else None})
        return audio, features


def decode_audio(source, timeout=120):
    if not source.get('has_audio'):
        return None
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', source['path'], '-vn',
        '-af', 'aresample=32000:async=1:first_pts=0', '-ac', '1', '-ar', '32000',
        '-f', 'f32le', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, check=True)
    return np.frombuffer(result.stdout, dtype='<f4').copy()


def measure_audio(samples, sample_rate=32000, step=.25):
    if samples is None or not len(samples):
        return {'status': 'unknown', 'windows': [], 'background_db': None}
    size = round(sample_rate*step)
    chunks = [samples[i:i+size] for i in range(0, len(samples), size)]
    db = np.array([20*np.log10(max(1e-8, float(np.sqrt(np.mean(c*c))))) for c in chunks])
    background = float(np.median(db))
    spread = max(3., float(np.median(abs(db-background)))*1.4826)
    return {'status': 'silent' if float(np.max(db)) < -70 else 'measured',
        'background_db': background, 'spread_db': spread,
        'windows': [{'start_sec': i*step, 'end_sec': min((i+1)*step, len(samples)/sample_rate),
            'relative_db': float(d-background), 'background_z': float((d-background)/spread),
            'transient_candidate': bool(i and d-db[i-1] > 9 and d > -60),
            'laughter': None, 'touch': None} for i, d in enumerate(db)]}


class LocalSoundDetector:
    """TorchScript SED adapter, e.g. an exported PANNs framewise model.

    The supplied export must accept mono float waveform [1, samples] at 32 kHz
    and return {'framewise_output': [1, time, classes]} probabilities. Labels
    come from the export's sidecar JSON in the exact training class order.
    Clipwise classifiers are deliberately rejected: they cannot locate events.
    No weights are downloaded, and no generic sound label is called a ball hit.
    """
    def __init__(self, model, labels):
        import torch
        if not isinstance(labels, list) or not labels or any(not isinstance(s, str) or not s for s in labels) or len(set(labels)) != len(labels):
            raise ValueError('声音模型类别必须为非空且不重复的字符串数组')
        self.torch = torch
        self.labels = labels
        self.model = torch.jit.load(str(Path(model)), map_location='cpu').eval()

    def detect(self, samples, start_sec=0., threshold=.5):
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(samples.copy())[None])
        values = output['framewise_output'].detach().cpu().numpy()
        if (values.ndim != 3 or values.shape[0] != 1 or values.shape[1] == 0 or values.shape[2] != len(self.labels)
                or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
            raise ValueError('声音事件模型输出不符合 framewise 概率协议')
        step = len(samples)/32000/values.shape[1]
        events = []
        for cls, label in enumerate(self.labels):
            hits = np.flatnonzero(values[0, :, cls] >= threshold)
            groups = np.split(hits, np.flatnonzero(np.diff(hits) > 1)+1)
            for group in groups:
                if len(group):
                    events.append({'label': label, 'start_sec': start_sec+int(group[0])*step,
                        'end_sec': start_sec+(int(group[-1])+1)*step,
                        'probability': float(values[0, group, cls].max()), 'association': 'unknown'})
        return events
