#!/usr/bin/env python3
"""Diagnostic for the already-installed YOLO action detector on original VNL labels.

The upstream training match list is not published with this checkpoint, so this
is not a leakage-free benchmark claim. No detection becomes a training label.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volleymole.action_spotting import CLASSES, file_hash
from volleymole.models import ModelRegistry

spec = importlib.util.spec_from_file_location('train_actions', ROOT/'scripts/train_action_spotter.py')
benchmark = importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--cache', type=Path, default=Path('runs/action_accuracy/yolo_cache'))
    parser.add_argument('--configuration', type=Path, default=Path('runs/action_accuracy/yolo_val_configuration.json'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--exclude-manifest', type=Path)
    args = parser.parse_args()
    began = time.monotonic()
    if args.report.exists():
        parser.error('This report already exists; preserve the frozen diagnostic result')
    split = benchmark.load_split(args.manifest, args.split)
    registry = ModelRegistry(); registry.verify(['action'])
    weights_hash = file_hash(registry.target('action'))
    if args.split == 'test':
        configuration = json.loads(args.configuration.read_text())
        if configuration['model_sha256'] != weights_hash or configuration['selected_on_split'] != 'val':
            raise ValueError('Freeze the same detector configuration on val first')
        if set(configuration['validation_matches']) & set(split['original_matches']):
            raise ValueError('Validation match overlaps test match')
        if args.exclude_manifest:
            old = json.loads(args.exclude_manifest.read_text())
            if {r['video'] for r in old['samples']} & {r['video'] for r in split['samples']}:
                raise ValueError('Previously inspected diagnostic clips overlap sealed test')
    from ultralytics import YOLO
    torch.set_num_threads(4); cv2.setNumThreads(1)
    model = YOLO(str(registry.target('action')))
    rows, probabilities = [], []
    args.cache.mkdir(parents=True, exist_ok=True)
    for sample in split['samples']:
        directory = split['root']/sample['frame_directory']
        paths = sorted(directory.glob('*.jpg'))
        if [p.name for p in paths] != [f'{i:06d}.jpg' for i in range(sample['frames'])]:
            raise ValueError('Source frame sequence is not complete')
        identity = {'source_frame_hash':sample['source_frame_hash_manifest_sha256'],
                    'weights_sha256':weights_hash, 'inference':{'imgsz':640, 'conf':.001, 'iou':.45}}
        import hashlib
        original_hash = hashlib.sha256()
        for path in paths:
            original_hash.update(f'{path.name} {file_hash(path)}\n'.encode())
        if original_hash.hexdigest() != identity['source_frame_hash']:
            raise ValueError('Original JPEG evidence changed')
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = args.cache/(signature+'.npy')
        if cache.exists():
            scores = np.load(cache, allow_pickle=False)
        else:
            scores = np.zeros((len(paths), len(CLASSES)), dtype=np.float32)
            for start in range(0, len(paths), 32):
                frames = [cv2.imread(str(path)) for path in paths[start:start+32]]
                results = model.predict(frames, device=args.device, imgsz=640, conf=.001, iou=.45,
                                        half=args.device.startswith('cuda'), verbose=False)
                for index, result in enumerate(results):
                    if result.boxes is None:
                        continue
                    for cls, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                        label = result.names[int(cls)]
                        if label in CLASSES:
                            column = CLASSES.index(label)
                            scores[start+index, column] = max(scores[start+index, column], float(conf))
            np.save(cache, scores, allow_pickle=False)
            benchmark.save_json(cache.with_suffix('.json'), identity)
        if scores.shape != (sample['frames'], len(CLASSES)) or not np.isfinite(scores).all():
            raise ValueError('Invalid cached detector evidence')
        probabilities.append(scores)
        rows.append({'sample':sample, 'truth':[{'label':e['label'], 'time_sec':e['frame']/sample['fps']}
                                             for e in sample['annotation_row']['events']]})
        print(f"YOLO {sample['video']} ({sample['frames']} frames)", flush=True)
    if args.split == 'val':
        selected = benchmark.choose_validation(rows, probabilities)
        configuration = {'selected_on_split':'val', 'model_sha256':weights_hash,
            'threshold':selected['threshold'], 'nms_sec':selected['nms_sec'],
            'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'validation_manifest_sha256':split['manifest_sha256'], 'validation_matches':split['original_matches'],
            'source_training_match_overlap':'unknown; diagnostic only', 'selection':selected}
        benchmark.save_json(args.configuration, configuration)
    metrics = {}
    for tolerance in (.2, .5, 1.):
        metrics[str(tolerance)], predictions = benchmark.score_rows(rows, probabilities,
            configuration['threshold'], configuration['nms_sec'], tolerance)
    report = {'benchmark':'existing action.pt original-label diagnostic', 'split':args.split,
        'complete':True, 'expected_clips':len(rows), 'completed_clips':len(rows), 'failures':[],
        'model_sha256':weights_hash, 'configuration':configuration, 'metrics':metrics,
        'class_names':model.names, 'predictions':predictions, 'new_annotations_created':False,
        'elapsed_sec':time.monotonic()-began, 'manifest_sha256':split['manifest_sha256'],
        'limitations':['Unknown original detector training match list; do not assert leakage-free generalization.',
                      'No score class in this checkpoint; all six-class metrics count missed original score events.',
                      'Frame object-action confidence is not a verified contact, outcome, or humor label.']}
    benchmark.save_json(args.report, report)
    print(json.dumps({'metrics':metrics, 'elapsed_sec':time.monotonic()-began}), flush=True)


if __name__ == '__main__': main()
