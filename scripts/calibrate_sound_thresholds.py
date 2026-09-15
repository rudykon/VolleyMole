#!/usr/bin/env python3
"""Select candidate thresholds using only fixed, originally labeled FSD50K dev/train.

No neural network is trained. ESC-50 is never used for threshold selection.
Its existing predictions may be re-scored afterward, explicitly as a nonblind
follow-up to the already observed baseline, not as a new held-out experiment.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_sound_events import binary_metrics, sha256, load_manifest
from prepare_sound_calibration import select_cohorts, TARGETS
from volleymole.audio_events import LocalSoundDetector, decode_audio


def select_f2_threshold(truth, scores):
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if (truth.ndim != 1 or truth.shape != scores.shape or not len(truth) or not truth.any()
            or truth.all() or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
        raise ValueError('Threshold fitting requires both classes and finite probabilities')
    best = None
    for threshold in sorted(set(scores.tolist()) | {1.}, reverse=True):
        predictions = scores >= threshold
        tp = int((predictions & truth).sum())
        fp = int((predictions & ~truth).sum())
        fn = int(truth.sum())-tp
        numerator, denominator = 5*tp, 5*tp+4*fn+fp
        # Integer comparison gives exact tie handling, independent of float rounding.
        if best is None or numerator*best['denominator'] > best['numerator']*denominator:
            best = {'threshold': float(threshold), 'numerator': numerator, 'denominator': denominator}
    return {'threshold': best['threshold'], 'f2': best['numerator']/best['denominator']}


def verified_manifest(dataset):
    manifest = json.loads((dataset/'manifest.json').read_text())
    if (manifest.get('dataset') != 'FSD50K' or not manifest.get('complete')
            or len(manifest['clips']) != manifest['expected_clips']):
        raise ValueError('Calibration requires the complete fixed FSD50K sample')
    protocol = manifest['protocol']
    if protocol['original_partition'] != 'dev/train' or protocol['new_annotations'] or protocol['network_training']:
        raise ValueError('Only original dev/train labels without new annotation are permitted')
    if sha256(dataset/'dev.csv') != protocol['dev_csv_sha256']:
        raise ValueError('Official dev labels changed')
    with (dataset/'dev.csv').open(newline='') as stream:
        original = list(csv.DictReader(stream))
    cohorts = select_cohorts(original, set(protocol['excluded_esc50_source_ids']))
    if cohorts != protocol['cohorts']:
        raise ValueError('Fixed pre-inference cohort selection changed')
    expected_ids = {item for cohort in cohorts.values() for items in cohort.values() for item in items}
    actual_ids = [r['fname'] for r in manifest['clips']]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError('Fixed sample is incomplete, duplicated or substituted')
    by_id = {r['fname']: r for r in original}
    for row in manifest['clips']:
        if row['split'] != 'train' or any(row[key] != by_id[row['fname']][key] for key in ('labels', 'mids', 'split')):
            raise ValueError('Original labels or split changed')
        if row['audio_path'] != f"audio/{row['fname']}.wav":
            raise ValueError('Unexpected calibration audio path')
        if sha256(dataset/row['audio_path']) != row['sha256']:
            raise ValueError('Calibration audio checksum differs from manifest')
    return manifest


def run(dataset, model, labels_path, output, sidecar_path, device='cuda:0', esc_baseline=None,
        esc_dataset=Path('data/public_benchmarks/audio/esc50')):
    started = time.monotonic()
    manifest = verified_manifest(dataset)
    labels = json.loads(labels_path.read_text())
    label_indices = {target: labels.index(target) for target in TARGETS}
    model_sha, labels_sha = sha256(model), sha256(labels_path)
    import torch
    torch.set_num_threads(4)
    identity = {'model_sha256': model_sha, 'labels_sha256': labels_sha,
                'manifest_sha256': sha256(dataset/'manifest.json'),
                'code_sha256': sha256(Path(__file__)), 'device': device,
                'adapter_code_sha256': sha256(ROOT/'src/volleymole/audio_events.py'),
                'torch_version': torch.__version__}
    detector = LocalSoundDetector(model, labels, device=device)
    predictions_path = output.with_suffix('.predictions.jsonl')
    identity_path = output.with_suffix('.identity.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if predictions_path.exists():
        if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
            raise ValueError('Calibration cache identity differs; use a new output path')
        for line in predictions_path.read_text().splitlines():
            row = json.loads(line)
            if row['fname'] in completed:
                raise ValueError('Duplicate cached prediction')
            completed[row['fname']] = row
    else:
        identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    rows = []
    with predictions_path.open('a') as stream:
        for index, row in enumerate(manifest['clips'], 1):
            result = completed.get(row['fname'])
            if result is None:
                samples = decode_audio({'path': str(dataset/row['audio_path']), 'has_audio': True})
                scores = detector.framewise(samples).max(axis=0)
                result = {key: row[key] for key in ('fname', 'labels', 'mids', 'split', 'sha256')}
                result['scores'] = {target: float(scores[cls]) for target, cls in label_indices.items()}
                stream.write(json.dumps(result, allow_nan=False)+'\n')
                stream.flush()
            elif any(result[key] != row[key] for key in ('labels', 'mids', 'split', 'sha256')):
                raise ValueError('Cached original clip provenance changed')
            rows.append(result)
            if index % 50 == 0:
                print(f'Calibration: {index}/{len(manifest["clips"])} clips inferred', flush=True)
    by_id = {r['fname']: r for r in rows}
    thresholds, metrics = {}, {}
    for target, cohort in manifest['protocol']['cohorts'].items():
        ids = cohort['positive_ids']+cohort['nominal_negative_ids']
        truth = [target in by_id[fname]['labels'].split(',') for fname in ids]
        scores = [by_id[fname]['scores'][target] for fname in ids]
        fit = select_f2_threshold(truth, scores)
        thresholds[target] = fit['threshold']
        metrics[target] = {'f2': fit['f2'], 'calibrated': binary_metrics(truth, scores, fit['threshold']),
                           'default_0_5': binary_metrics(truth, scores, .5)}
    if sha256(model) != model_sha or sha256(labels_path) != labels_sha:
        raise ValueError('Model or labels changed during calibration')
    if sha256(ROOT/'src/volleymole/audio_events.py') != identity['adapter_code_sha256']:
        raise ValueError('Production audio adapter changed during calibration')
    report = {'identity': identity, 'dataset': 'FSD50K original dev/train', 'unique_clips': len(rows),
              'protocol': manifest['protocol'], 'thresholds': thresholds, 'development_metrics': metrics,
              'network_training': False, 'new_annotations': False,
              'elapsed_sec': time.monotonic()-started,
              'limitations': ['These are detection thresholds, not calibrated probabilities.',
                              'Development F2 is fitted on the same fixed calibration cohort, not an independent estimate.',
                              'Single target absence in original weak positive labels is a nominal negative, not verified absence.',
                              'Clip-level fitting does not validate temporal localization or court association.',
                              'Sampling by numeric ID is reproducible but is not population-representative random sampling.']}
    if esc_baseline is not None:
        baseline = json.loads(esc_baseline.read_text())
        if (baseline['identity']['model_sha256'] != model_sha or baseline['identity']['labels_sha256'] != labels_sha
                or not baseline['complete']):
            raise ValueError('ESC baseline must contain complete predictions from this exact model')
        if json.loads(esc_baseline.with_suffix('.identity.json').read_text()) != baseline['identity']:
            raise ValueError('ESC prediction identity differs from baseline report')
        esc_manifest = load_manifest(esc_dataset)
        if sha256(esc_dataset/'manifest.json') != baseline['identity']['manifest_sha256']:
            raise ValueError('ESC dataset differs from the baseline manifest')
        esc_rows = [json.loads(line) for line in esc_baseline.with_suffix('.predictions.jsonl').read_text().splitlines()]
        if len(esc_rows) != 2000 or len({r['filename'] for r in esc_rows}) != 2000:
            raise ValueError('ESC baseline prediction coverage changed')
        original_esc = {row['filename']: row for row in esc_manifest['clips']}
        if set(original_esc) != {row['filename'] for row in esc_rows}:
            raise ValueError('ESC prediction file identities changed')
        for row in esc_rows:
            if any(row[key] != original_esc[row['filename']][key] for key in ('category', 'fold', 'src_file', 'sha256')):
                raise ValueError('ESC cached labels or audio provenance differ from the original manifest')
        report['esc50_followup'] = {
            'status': 'Nonblind follow-up: ESC baseline was already observed; no ESC labels used to choose thresholds.',
            'baseline_report_sha256': sha256(esc_baseline), 'reused_predictions_sha256': sha256(esc_baseline.with_suffix('.predictions.jsonl')),
            'clips': 2000, 'metrics': {
                category: binary_metrics([r['category'] == category for r in esc_rows],
                                         [r['scores'][category] for r in esc_rows], thresholds[target])
                for category, target in [('laughing', 'Laughter'), ('clapping', 'Clapping')]},
            'cheering': None}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    sidecar = {'schema_version': 1, 'model_sha256': model_sha, 'labels_sha256': labels_sha, 'thresholds': thresholds,
               'calibration': {'dataset': 'FSD50K', 'original_partition': 'dev/train',
                               'source_url': 'https://zenodo.org/records/4060432',
                               'protocol': manifest['protocol']['protocol'],
                               'manifest_sha256': identity['manifest_sha256'],
                               'report': str(output), 'report_sha256': sha256(output),
                               'unique_clips': len(rows), 'new_annotations': False, 'network_training': False,
                               'esc50_used_for_threshold_selection': False,
                               'purpose': 'Candidate discovery only; not calibrated probability or temporal/visual ground truth'}}
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(sidecar_path)
    print(json.dumps({'thresholds': thresholds, 'development_metrics': metrics,
                      'esc50_followup': report.get('esc50_followup')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/public_benchmarks/audio/fsd50k_dev_calibration'))
    parser.add_argument('--model', type=Path, default=Path('models/audio/panns_cnn14_sed.pt'))
    parser.add_argument('--labels', type=Path, default=Path('models/audio/panns_cnn14_sed.labels.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/public_benchmarks/fsd50k_threshold_calibration.json'))
    parser.add_argument('--sidecar', type=Path, default=Path('models/audio/panns_cnn14_sed.thresholds.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--esc-baseline', type=Path, default=Path('runs/public_benchmarks/esc50_panns.json'))
    parser.add_argument('--esc-dataset', type=Path, default=Path('data/public_benchmarks/audio/esc50'))
    args = parser.parse_args()
    run(args.dataset, args.model, args.labels, args.output, args.sidecar, args.device, args.esc_baseline, args.esc_dataset)


if __name__ == '__main__':
    main()
