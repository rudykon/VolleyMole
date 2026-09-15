#!/usr/bin/env python3
"""Evaluate deployed SED against original ESC-50 clip labels, without tuning.

All 2,000 original clips and their five official folds are retained.
The framewise detector is max-pooled only for this clip-level benchmark.
ESC-50 has no strong timestamps, cheering class, or speech class, so this
evaluation cannot establish event localization or in-match association quality.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from volleymole.audio_events import LocalSoundDetector, decode_audio

LABEL_MAP = {'laughing': ['Laughter'], 'clapping': ['Clapping']}
PROTOCOL = 'esc50-official-all-folds-max-framewise-v1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binary_metrics(truth, scores, threshold=.5):
    """Non-interpolated AP and rank AUROC with equal-score ties grouped exactly."""
    truth = np.asarray(truth)
    scores = np.asarray(scores, dtype=float)
    if (truth.ndim != 1 or scores.shape != truth.shape or not np.isin(truth, [0, 1]).all()
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or not 0 <= threshold <= 1):
        raise ValueError('Binary labels and finite probabilities must align')
    truth = truth.astype(bool)
    positive, negative = int(truth.sum()), int((~truth).sum())
    prediction = scores >= threshold
    tp, fp = int((truth & prediction).sum()), int((~truth & prediction).sum())
    fn, tn = positive-tp, negative-fp
    ap = auc = None
    if positive and negative:
        order = np.argsort(-scores, kind='stable')
        sorted_truth, sorted_scores = truth[order], scores[order]
        boundaries = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
        cumulative_tp = np.cumsum(sorted_truth)[boundaries]
        cumulative_fp = boundaries+1-cumulative_tp
        recall = cumulative_tp/positive
        precision = cumulative_tp/(boundaries+1)
        ap = float(np.sum(np.diff(np.r_[0., recall])*precision))
        auc = float(np.trapezoid(np.r_[0., recall], np.r_[0., cumulative_fp/negative]))
    return {'clips': len(truth), 'positive_clips': positive, 'nominal_negative_clips': negative,
            'average_precision': ap, 'auroc': auc, 'fixed_threshold': threshold,
            'true_positive': tp, 'false_positive_vs_single_class_labels': fp,
            'false_negative': fn, 'true_negative_vs_single_class_labels': tn,
            'precision_vs_single_class_labels': tp/(tp+fp) if tp+fp else None,
            'recall': tp/positive if positive else None,
            'nominal_false_positive_rate': fp/negative if negative else None}


def clip_scores(detector, samples):
    """Use the production adapter and retain low framewise scores before pooling."""
    values = detector.framewise(samples)
    scores = values.max(axis=0)
    probabilities = dict(zip(detector.labels, scores, strict=True))
    return {category: float(max(probabilities[label] for label in labels)) for category, labels in LABEL_MAP.items()}


def summarize(rows, threshold=.5):
    groups = {'all_original_folds': rows}
    groups.update({f'fold_{fold}': [row for row in rows if row['fold'] == fold] for fold in range(1, 6)})
    report = {}
    for name, clips in groups.items():
        report[name] = {category: binary_metrics([r['category'] == category for r in clips],
                                                [r['scores'][category] for r in clips], threshold)
                        for category in LABEL_MAP}
    report['background_label_disagreements'] = {
        category: dict(Counter(r['category'] for r in rows
                               if r['category'] != category and r['scores'][category] >= threshold))
        for category in LABEL_MAP}
    return report


def load_manifest(dataset):
    manifest_path = dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('dataset') != 'ESC-50' or len(manifest.get('clips', [])) != 2000:
        raise ValueError('Use complete official ESC-50 from prepare_public_audio.py')
    if sha256(dataset/manifest['annotation_file']) != manifest['annotation_sha256']:
        raise ValueError('Original annotation checksum mismatch')
    with (dataset/manifest['annotation_file']).open(newline='') as stream:
        original = list(csv.DictReader(stream))
    by_name = {r['filename']: r for r in original}
    if len(by_name) != 2000 or len({r['filename'] for r in manifest['clips']}) != 2000:
        raise ValueError('Official clip coverage must be unique and complete')
    source_folds = {}
    for row in manifest['clips']:
        source = by_name.get(row['filename'], {})
        for key in ('filename', 'fold', 'target', 'category', 'src_file', 'take'):
            if str(row[key]) != str(source.get(key)):
                raise ValueError('Manifest changes an original annotation')
        if row['audio_path'] != f"audio/{row['filename']}" or Path(row['filename']).name != row['filename']:
            raise ValueError('Invalid audio path')
        source_folds.setdefault(row['src_file'], set()).add(row['fold'])
    overlap = {source: sorted(folds) for source, folds in source_folds.items() if len(folds) > 1}
    if overlap != manifest.get('original_cross_fold_source_ids', {}):
        raise ValueError('Original source-fold overlap audit differs from manifest')
    return manifest


def evaluate(dataset, model, labels_path, output, device='cpu', threads=4):
    started = time.monotonic()
    manifest = load_manifest(dataset)
    labels = json.loads(labels_path.read_text())
    missing = sorted({s for v in LABEL_MAP.values() for s in v}-set(labels))
    if missing:
        raise ValueError(f'Model classes required by fixed mapping are missing: {missing}')
    import torch
    torch.set_num_threads(threads)
    detector = LocalSoundDetector(model, labels, device=device)
    identity = {'protocol': PROTOCOL, 'model_sha256': sha256(model),
                'labels_sha256': sha256(labels_path), 'manifest_sha256': sha256(dataset/'manifest.json'),
                'evaluation_code_sha256': sha256(Path(__file__)),
                'adapter_code_sha256': sha256(Path(__file__).resolve().parents[1]/'src/volleymole/audio_events.py'),
                'label_map': LABEL_MAP, 'device': device, 'torch_version': torch.__version__}
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.with_suffix('.predictions.jsonl')
    identity_path = output.with_suffix('.identity.json')
    completed = {}
    if predictions_path.exists():
        if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
            raise ValueError('Prediction cache belongs to another model, dataset or evaluator; use a new --output')
        for line in predictions_path.read_text().splitlines():
            row = json.loads(line)
            if row['filename'] in completed:
                raise ValueError('Duplicate clip in prediction cache')
            completed[row['filename']] = row
    else:
        identity_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False)+'\n')
    failures, result_rows, fresh_clips = [], [], 0
    with predictions_path.open('a') as out:
        for index, row in enumerate(manifest['clips']):
            audio_path = dataset/row['audio_path']
            if not audio_path.is_file() or sha256(audio_path) != row['sha256']:
                failures.append({'filename': row['filename'], 'error': 'Missing or changed original audio'})
                continue
            cached = completed.get(row['filename'])
            if cached is not None:
                if any(cached[key] != row[key] for key in ('sha256', 'fold', 'category', 'src_file')):
                    raise ValueError('Cached clip provenance differs from original manifest')
                result_rows.append(cached)
                continue
            try:
                before = time.monotonic()
                samples = decode_audio({'path': str(audio_path), 'has_audio': True})
                scores = clip_scores(detector, samples)
                prediction = {key: row[key] for key in ('filename', 'fold', 'category', 'src_file', 'sha256')}
                prediction.update(scores=scores, decode_and_inference_sec=time.monotonic()-before)
                out.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False)+'\n')
                out.flush()
                result_rows.append(prediction)
                fresh_clips += 1
            except Exception as exc:
                failures.append({'filename': row['filename'], 'error': f'{type(exc).__name__}: {exc}'})
            if (index+1) % 100 == 0:
                print(f'{index+1}/2000 clips; {len(failures)} failures; {time.monotonic()-started:.1f}s', flush=True)
    report = {'identity': identity, 'dataset': 'ESC-50', 'expected_clips': 2000,
              'evaluated_clips': len(result_rows), 'complete': len(result_rows) == 2000 and not failures,
              'original_cross_fold_source_ids': manifest.get('original_cross_fold_source_ids', {}),
              'freshly_inferred_clips': fresh_clips, 'reused_clips': len(result_rows)-fresh_clips,
              'failures': failures, 'metrics': summarize(result_rows), 'threshold_selected_on_this_dataset': False,
              'training_on_this_dataset': False, 'event_localization_metrics': None,
              'cheering_metrics': None, 'speech_metrics': None, 'volleyball_sound_association_metrics': None,
              'elapsed_this_invocation_sec': time.monotonic()-started,
              'decode_and_inference_sec': sum(r['decode_and_inference_sec'] for r in result_rows),
              'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                           'device': device, 'cpu_threads': threads,
                           'gpu': torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None},
              'limitations': [
                  'Original human single-class labels are reused verbatim; no new labels or pseudo labels.',
                  'Other ESC-50 categories are nominal negatives; background target sounds are not exhaustively annotated.',
                  'AP/AUROC and fixed 0.5 threshold counts describe clip classification, not temporal event detection.',
                  'No cheering or speech positives, strong timestamps, court association, or volleyball outcomes are annotated.',
                  'Model was not trained or calibrated here; exact overlap with original model pretraining has not been audited.',
                  'All original folds are evaluation partitions here, not a newly trained five-fold cross-validation result.',
                  'Four original Freesound source IDs cross official folds; they are recorded without silently altering the official split.',
                  'Elapsed time is audio evaluation only; it is not full VolleyMole runtime or the 25% performance target.',
              ]}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/public_benchmarks/audio/esc50'))
    parser.add_argument('--model', type=Path, default=Path('models/audio/panns_cnn14_sed.pt'))
    parser.add_argument('--labels', type=Path, default=Path('models/audio/panns_cnn14_sed.labels.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/public_benchmarks/esc50_panns.json'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    report = evaluate(args.dataset, args.model, args.labels, args.output, args.device, args.threads)
    print(json.dumps(report['metrics']['all_original_folds'], indent=2))
    if not report['complete']:
        raise SystemExit('Incomplete benchmark; inspect failures in report')


if __name__ == '__main__':
    main()
