#!/usr/bin/env python3
"""Install published PANNs SED weights, with pinned hashes and offline export.

Run with the project's Python environment. No training, data annotation,
torchlibrosa dependency or remote Python code execution is involved.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

CHECKPOINT_NAME = 'Cnn14_DecisionLevelMax_mAP=0.385.pth'
CHECKPOINT_URL = ('https://zenodo.org/records/3987831/files/'
                  'Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1')
CHECKPOINT_MD5 = '70539c43c18b6a289b3199c503a82c5a'
CHECKPOINT_SHA256 = 'dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142'
LABELS_URL = 'https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
LABELS_SHA256 = 'cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429'


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download_verified(url, path, sha256, no_proxy=False):
    path = Path(path)
    if path.is_file() and digest(path) == sha256:
        return path
    opener = urllib.request.build_opener(*([urllib.request.ProxyHandler({})] if no_proxy else []))
    temporary = path.with_suffix(path.suffix + '.download')
    request = urllib.request.Request(url, headers={'User-Agent': 'VolleyMole/0.2 public-model-install'})
    print(f'Downloading {url}', flush=True)
    try:
        with opener.open(request, timeout=60) as response, temporary.open('wb') as target:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                target.write(block)
        if digest(temporary) != sha256:
            raise ValueError(f'Download checksum mismatch: {path.name}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'models/audio')
    parser.add_argument('--checkpoint', type=Path, help='Use an existing exact published checkpoint (still hash checked)')
    parser.add_argument('--checkpoint-source', help='Record the retrieval URL when an existing checkpoint came from a mirror')
    parser.add_argument('--labels-csv', type=Path, help='Use an existing exact official AudioSet CSV (still hash checked)')
    parser.add_argument('--no-proxy', action='store_true', help='Use direct network access when the configured proxy is unavailable')
    parser.add_argument('--verify-audio', type=Path, help='Verify variable-length export on a real local audio recording')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or download_verified(CHECKPOINT_URL, args.directory / CHECKPOINT_NAME,
                                                      CHECKPOINT_SHA256, args.no_proxy)
    if digest(checkpoint) != CHECKPOINT_SHA256 or digest(checkpoint, 'md5') != CHECKPOINT_MD5:
        raise ValueError('Checkpoint differs from the published PANNs Cnn14_DecisionLevelMax weights')
    csv_path = args.labels_csv or download_verified(LABELS_URL, args.directory / 'class_labels_indices.csv',
                                                   LABELS_SHA256, args.no_proxy)
    if digest(csv_path) != LABELS_SHA256:
        raise ValueError('AudioSet class order CSV differs from the pinned official file')
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [int(row['index']) for row in rows] != list(range(527)):
        raise ValueError('Invalid AudioSet class indices')
    labels = [row['display_name'] for row in rows]
    import torch
    from volleymole.panns import PannsCnn14Sed
    torch.set_num_threads(2)
    # Loading only tensors avoids arbitrary pickle globals in old checkpoints.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = PannsCnn14Sed().eval()
    model.load_state_dict(state['model'], strict=True)
    scripted = torch.jit.script(model)
    stem = args.directory / 'panns_cnn14_sed'
    label_path = stem.with_suffix('.labels.json')
    label_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + '\n')
    provenance = {
        'model': 'PANNs Cnn14_DecisionLevelMax', 'checkpoint_url': CHECKPOINT_URL,
        'retrieved_from': args.checkpoint_source or (str(checkpoint) if args.checkpoint else CHECKPOINT_URL),
        'source_record': 'https://zenodo.org/records/3987831',
        'checkpoint_md5': CHECKPOINT_MD5, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'weights_license': 'CC-BY-4.0', 'code_license': 'MIT',
        'authors': ['Qiuqiang Kong', 'Yin Cao', 'Turab Iqbal', 'Yuxuan Wang', 'Wenwu Wang', 'Mark D. Plumbley'],
        'paper': 'https://arxiv.org/abs/1912.10211',
        'architecture_source': 'https://github.com/qiuqiangkong/audioset_tagging_cnn',
        'training_dataset': 'AudioSet, 527 classes, existing weak clip labels',
        'local_training': False, 'new_annotations': False,
        'labels_url': LABELS_URL, 'labels_csv_sha256': LABELS_SHA256,
        'labels_json_sha256': digest(label_path),
        'sample_rate': 32000, 'frame_hop_samples': 320, 'independent_decision_hop_samples': 10240,
        'short_input_policy': 'Zero-pad below 320 ms internally; crop output to original duration.',
        'temporal_limit': '10 ms output is interpolated from 320 ms CNN steps; not ball-contact timing evidence.',
        'association_limit': 'Labels alone do not identify the court or justify highlight/blooper scores.',
        'export': 'torch.jit.script, dynamic input length, strict published state_dict',
        'torch_version': torch.__version__, 'architecture_sha256': digest(ROOT / 'src/volleymole/panns.py'),
        'installed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    model_path = stem.with_suffix('.pt')
    temp_model = model_path.with_suffix('.pt.tmp')
    torch.jit.save(scripted, str(temp_model), _extra_files={'provenance.json': json.dumps(provenance)})
    provenance['export_sha256'] = digest(temp_model)
    if args.verify_audio:
        import numpy as np
        from volleymole.audio_events import decode_audio, LocalSoundDetector
        audio = decode_audio({'has_audio': True, 'path': str(args.verify_audio)})
        if audio is None or len(audio) < 32000:
            raise ValueError('Verification requires at least one second of real audio')
        detector = LocalSoundDetector(temp_model, labels)
        checks = []
        for seconds in (0.08, 0.32, 1.25, 2.73, 5.0, len(audio) / 32000):
            sample = audio[:min(round(seconds * 32000), len(audio))]
            actual = detector.framewise(sample)
            with torch.inference_mode():
                expected = model(torch.from_numpy(sample.copy())[None])['framewise_output'][0].numpy()
            error = float(np.max(np.abs(actual - expected)))
            if actual.shape != (len(sample) // 320 + 1, 527) or error > 1e-5:
                raise ValueError('Scripted export failed dynamic real-audio equivalence')
            events = detector.detect(sample, start_sec=13.25, threshold=0.)
            if any(e['start_sec'] < 13.25 or e['end_sec'] > 13.25 + len(sample) / 32000 + 1e-8 for e in events):
                raise ValueError('Detected event falls outside source audio')
            checks.append({'seconds': len(sample) / 32000, 'shape': list(actual.shape),
                           'script_eager_max_abs_error': error})
        provenance['real_audio_verification'] = {'path': str(args.verify_audio),
            'sha256': digest(args.verify_audio), 'purpose': 'export and timing protocol only; not accuracy', 'checks': checks}
    stem.with_suffix('.manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    # Publish the model last so automatic discovery never observes an export
    # while its requested real-audio verification is still pending.
    temp_model.replace(model_path)
    print(json.dumps({'model': str(model_path), 'labels': str(label_path),
                      'manifest': str(stem.with_suffix('.manifest.json')), 'sha256': provenance['export_sha256']}, indent=2))


if __name__ == '__main__':
    main()
