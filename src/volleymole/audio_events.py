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


def validate_sound_thresholds(thresholds, labels):
    """Validate detection cutoffs; these do not calibrate probabilities."""
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError('声音阈值必须为非空的类别到数值映射')
    if any(label not in labels for label in thresholds):
        raise ValueError('声音阈值包含模型类别中不存在的名称')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError('声音阈值必须为 [0, 1] 范围内的有限数值')
    return {label: float(value) for label, value in thresholds.items()}


def load_sound_thresholds(path, model, labels_path):
    """Read a sidecar only when it is bound to these exact model/label bytes."""
    from .common import digest, read_json
    data = read_json(path)
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('声音阈值文件 schema_version 必须为 1')
    if data.get('model_sha256') != digest(model) or data.get('labels_sha256') != digest(labels_path):
        raise ValueError('声音阈值文件与当前模型或类别文件的 SHA256 不匹配')
    labels = read_json(labels_path)
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) for label in labels):
        raise ValueError('声音类别文件必须为非空字符串数组')
    return validate_sound_thresholds(data.get('thresholds'), labels)


class LocalSoundDetector:
    """TorchScript SED adapter, e.g. an exported PANNs framewise model.

    The supplied export must accept mono float waveform [1, samples] at 32 kHz
    and return {'framewise_output': [1, time, classes]} probabilities. Labels
    come from the export's sidecar JSON in the exact training class order.
    Clipwise classifiers are deliberately rejected: they cannot locate events.
    Installation is explicit (scripts/install_sound_model.py); inference never
    downloads files, and no generic sound label is called a ball hit.
    """
    def __init__(self, model, labels, device='cpu', thresholds=None):
        import torch
        if not isinstance(labels, list) or not labels or any(not isinstance(s, str) or not s for s in labels) or len(set(labels)) != len(labels):
            raise ValueError('声音模型类别必须为非空且不重复的字符串数组')
        self.torch = torch
        self.labels = labels
        self.thresholds = validate_sound_thresholds(thresholds, labels) if thresholds is not None else {}
        self.device = torch.device(device)
        self.model = torch.jit.load(str(Path(model)), map_location=self.device).eval()

    def framewise(self, samples):
        """Return measured probabilities [time, class], including low scores.

        This interface also supports evaluation against existing dataset labels
        without turning predictions into new ground-truth annotations.
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError('声音模型输入必须为非空、有限值的单声道波形')
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(samples.copy())[None].to(self.device))
        if not isinstance(output, dict) or 'framewise_output' not in output:
            raise ValueError('声音事件模型必须返回 framewise_output，不能使用整段分类输出代替')
        values = output['framewise_output'].detach().cpu().numpy()
        if (values.ndim != 3 or values.shape[0] != 1 or values.shape[1] == 0 or values.shape[2] != len(self.labels)
                or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
            raise ValueError('声音事件模型输出不符合 framewise 概率协议')
        return values[0]

    def detect(self, samples, start_sec=0., threshold=.5):
        if not np.isfinite(start_sec) or start_sec < 0 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('声音事件起点和概率阈值无效')
        values = self.framewise(samples)
        duration = len(samples)/32000
        # PANNs exports declare their 10 ms output grid. Its independent CNN
        # decisions are only 320 ms apart: interpolation adds no new evidence.
        hop = getattr(self.model, 'frame_hop_samples', None)
        step = hop/32000 if hop is not None else duration/values.shape[0]
        events = []
        for cls, label in enumerate(self.labels):
            # The scalar remains the fallback for uncalibrated classes.
            # Returned probability is the original network output, unchanged.
            hits = np.flatnonzero(values[:, cls] >= self.thresholds.get(label, threshold))
            groups = np.split(hits, np.flatnonzero(np.diff(hits) > 1)+1)
            for group in groups:
                if len(group):
                    start = min(duration, int(group[0])*step)
                    end = min(duration, (int(group[-1])+1)*step)
                    if end <= start:
                        continue
                    events.append({'label': label, 'start_sec': start_sec+start,
                        'end_sec': start_sec+end,
                        'probability': float(values[group, cls].max()), 'association': 'unknown'})
        return events
