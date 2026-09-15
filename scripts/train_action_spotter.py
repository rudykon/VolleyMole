#!/usr/bin/env python3
"""Train local VNL action proposals on author train labels; select on val only.

An explicit evaluate command consumes the frozen model and a held-out manifest.
No VLM prediction, test label, or new annotation enters training or calibration.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volleymole.action_spotting import (BACKBONE_URL, CLASSES, FEATURE_DIM, FEATURE_VERSION,
    FrozenRGB, TemporalHead, dense_probabilities, file_hash, proposals, soft_targets)

spec = importlib.util.spec_from_file_location('original_action_evaluator', ROOT/'scripts/evaluate_volleyball_actions.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def load_split(manifest_path, split):
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest['dataset'] != 'VNL-STES' or split not in ('train', 'val', 'test'):
        raise ValueError('Unknown source dataset or split')
    # Only this split is read. Training has no dependency on original test labels.
    original_path = root/f'raw/vnl_1.5/{split}.json'
    originals = json.loads(original_path.read_text())
    indexed = {row['video']:row for row in originals}
    source_artifact = next(row for row in manifest['original_annotation_files']
                           if row['path'] == f'raw/vnl_1.5/{split}.json')
    if file_hash(original_path) != source_artifact['sha256']:
        raise ValueError('Original split annotation hash changed')
    samples = manifest['samples']
    if not samples or len({s['video'] for s in samples}) != len(samples):
        raise ValueError('Empty or duplicate sample coverage')
    for sample in samples:
        if (sample['split'] != split or sample['annotation_row'] != indexed.get(sample['video'])
                or sample['match_id'] != sample['video'].split('/')[0]
                or sample['fps'] != 25 or sample['annotation_row']['fps'] != 25
                or sample['first_frame_file'] != '000000.jpg'):
            raise ValueError('Sample differs from original split, ontology, or 25 FPS clock')
        if sample['source_declared_num_frames'] != sample['annotation_row']['num_frames']:
            raise ValueError('Original declared frame count changed')
        if (sample['frames'] < sample['source_declared_num_frames'] or
                sample.get('frame_count_matches_annotation', sample['frames'] == sample['source_declared_num_frames'])
                != (sample['frames'] == sample['source_declared_num_frames'])):
            raise ValueError('Source sequence is truncated or frame discrepancy is hidden')
        soft_targets(sample['annotation_row']['events'], sample['frames'], sample['fps'])
    return {'root':root, 'manifest':manifest, 'samples':samples,
            'manifest_sha256':file_hash(manifest_path), 'original_split_sha256':file_hash(original_path),
            'original_matches':sorted({row['video'].split('/')[0] for row in originals}), 'split':split}


def check_training_separation(train, val):
    if train['split'] != 'train' or val['split'] != 'val':
        raise ValueError('Only original train and val splits can fit/select a model')
    if set(train['original_matches']) & set(val['original_matches']):
        raise ValueError('Original match identity leaks across train and val')
    if {r['video'] for r in train['samples']} & {r['video'] for r in val['samples']}:
        raise ValueError('Duplicate clips across train and validation')


def features_for(split, encoder, cache, batch_size):
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sample in split['samples']:
            directory = split['root']/sample['frame_directory']
            paths = sorted(directory.glob('*.jpg'))
            expected = [f'{i:06d}.jpg' for i in range(sample['frames'])]
            if [p.name for p in paths] != expected:
                raise ValueError(f"Nonconsecutive source frames: {sample['video']}")
            hashed = hashlib.sha256()
            for path in paths:
                hashed.update(f'{path.name} {file_hash(path)}\n'.encode())
            if hashed.hexdigest() != sample['source_frame_hash_manifest_sha256']:
                raise ValueError('Original JPEG evidence hash changed')
            identity = {'source':hashed.hexdigest(), 'encoder':encoder.sha256,
                        'version':FEATURE_VERSION, 'compute_policy':encoder.compute_policy,
                        'code':file_hash(ROOT/'src/volleymole/action_spotting.py')}
            signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            feature_file = cache/(signature+'.npy')
            if feature_file.is_file():
                features = np.load(feature_file, allow_pickle=False).astype(np.float32)
            else:
                batches = []
                for start in range(0, len(paths), batch_size):
                    frames = list(pool.map(lambda p: cv2.imread(str(p)), paths[start:start+batch_size]))
                    batches.append(encoder.encode(frames, batch_size))
                features = np.concatenate(batches)
                np.save(feature_file, features.astype(np.float16), allow_pickle=False)
                # Train and cached runs use exactly the same rounded features.
                features = features.astype(np.float16).astype(np.float32)
                save_json(feature_file.with_suffix('.json'), identity)
            if features.shape != (sample['frames'], FEATURE_DIM) or not np.isfinite(features).all():
                raise ValueError('Invalid cached RGB features')
            result.append({'sample':sample, 'features':features,
                'targets':soft_targets(sample['annotation_row']['events'], sample['frames'], sample['fps']),
                'truth':[{'label':e['label'], 'time_sec':e['frame']/sample['fps']}
                         for e in sample['annotation_row']['events']]})
            print(f"Features {split['split']} {sample['video']} ({len(features)} frames)", flush=True)
    return result


def score_rows(rows, probabilities, threshold, nms_sec, tolerance):
    counts = {label:{'tp':0, 'fp':0, 'fn':0} for label in CLASSES}
    predictions = []
    for row, scores in zip(rows, probabilities, strict=True):
        sample = row['sample']
        events = proposals(scores, np.arange(len(scores))/sample['fps'], threshold, nms_sec)
        predictions.append({'video':sample['video'], 'events':events})
        for label, value in evaluator.match_counts(row['truth'], events, tolerance).items():
            for key, number in value.items():
                counts[label][key] += number
    return {'micro':evaluator.metrics({key:sum(v[key] for v in counts.values()) for key in ('tp', 'fp', 'fn')}),
            'per_class':{label:evaluator.metrics(value) for label, value in counts.items()}}, predictions


def choose_validation(rows, probabilities):
    candidates = []
    # Predeclared shared thresholds and NMS grid, never chosen per test clip.
    for threshold in (.05, .1, .2, .3, .4, .5, .6, .7, .8):
        for nms_sec in (.12, .24, .4):
            metrics, _ = score_rows(rows, probabilities, threshold, nms_sec, .5)
            candidates.append({'threshold':threshold, 'nms_sec':nms_sec, 'metrics':metrics})
    # Precision then stricter threshold resolve equal-F1 settings deterministically.
    return max(candidates, key=lambda r: (r['metrics']['micro']['f1'] or 0,
                 r['metrics']['micro']['precision'] or 0, r['threshold'], r['nms_sec']))


def make_batch(rows, rng, batch_size, clip_len):
    indexes = list(range(len(rows)))
    rng.shuffle(indexes)
    # Every original training rally contributes each epoch, including short rallies.
    clips = []
    for index in indexes:
        row = rows[index]
        for _ in range(max(1, math.ceil(len(row['features'])/clip_len))):
            start = rng.randint(0, max(0, len(row['features'])-clip_len))
            clips.append((row, start))
    rng.shuffle(clips)
    for start in range(0, len(clips), batch_size):
        group = clips[start:start+batch_size]
        x = np.zeros((len(group), clip_len, FEATURE_DIM), dtype=np.float32)
        y = np.zeros((len(group), clip_len, len(CLASSES)), dtype=np.float32)
        mask = np.zeros((len(group), clip_len, 1), dtype=np.float32)
        lengths = []
        for i, (row, offset) in enumerate(group):
            length = min(clip_len, len(row['features'])-offset)
            x[i, :length] = row['features'][offset:offset+length]
            y[i, :length] = row['targets'][offset:offset+length]
            mask[i, :length] = 1
            lengths.append(length)
        yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.tensor(lengths)


def train(args):
    began = time.monotonic()
    train_split = load_split(args.train_manifest, 'train')
    val_split = load_split(args.val_manifest, 'val')
    check_training_separation(train_split, val_split)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    encoder = FrozenRGB(args.backbone).to(args.device).eval()
    feature_compute_policy = encoder.compute_policy
    train_rows = features_for(train_split, encoder, args.cache, args.feature_batch)
    val_rows = features_for(val_split, encoder, args.cache, args.feature_batch)
    del encoder
    model = TemporalHead().to(args.device)
    feature_values = np.concatenate([r['features'] for r in train_rows])
    model.feature_mean.copy_(torch.from_numpy(feature_values.mean(0)).to(args.device))
    model.feature_scale.copy_(torch.from_numpy(np.maximum(feature_values.std(0), .05)).to(args.device))
    del feature_values
    targets = np.concatenate([r['targets'] for r in train_rows])
    # Estimated only from original training labels, no test-derived class prior.
    pos_weight = torch.tensor(np.clip(np.sqrt((len(targets)-targets.sum(0))/(targets.sum(0)+1)), 1, 20),
                              dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    rng = random.Random(args.seed)
    best, best_state, history = None, None, []
    stale = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train(); total, batches = 0., 0
        for x, y, mask, lengths in make_batch(train_rows, rng, args.batch_size, 128):
            x, y, mask = (item.to(args.device) for item in (x, y, mask))
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = (F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction='none')*mask).sum()/(mask.sum()*len(CLASSES))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step(); total += float(loss.detach()); batches += 1
        probabilities = [dense_probabilities(model, row['features']) for row in val_rows]
        selected = choose_validation(val_rows, probabilities)
        row = {'epoch':epoch, 'train_loss':total/batches, 'selection':selected}
        history.append(row)
        current_key = (selected['metrics']['micro']['f1'] or 0, selected['metrics']['micro']['precision'] or 0)
        best_key = (-1, -1) if best is None else (best['selection']['metrics']['micro']['f1'] or 0,
                                                best['selection']['metrics']['micro']['precision'] or 0)
        if current_key > best_key:
            best, best_state, stale = copy.deepcopy(row), {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            stale += 1
        print(f"Epoch {epoch}: loss={total/batches:.5f} val F1={current_key[0]:.4f} "
              f"threshold={selected['threshold']} nms={selected['nms_sec']}", flush=True)
        if stale >= args.patience:
            break
    temporary = args.output.with_suffix('.pt.part')
    torch.save(best_state, temporary); temporary.replace(args.output)
    config = {'fps':25., 'window':192, 'stride':96,
              'threshold':best['selection']['threshold'], 'nms_sec':best['selection']['nms_sec']}
    manifest = {'schema_version':1, 'architecture':'frozen torchvision ResNet18 + two-layer bidirectional GRU',
        'classes':list(CLASSES), 'feature_version':FEATURE_VERSION, 'checkpoint_sha256':file_hash(args.output),
        'backbone_sha256':file_hash(args.backbone), 'backbone_url':BACKBONE_URL,
        'training_feature_compute_policy':feature_compute_policy,
        'created_at_utc':datetime.now(timezone.utc).isoformat(), 'inference':config,
        'training_manifest_sha256':train_split['manifest_sha256'], 'validation_manifest_sha256':val_split['manifest_sha256'],
        'training_original_annotation_sha256':train_split['original_split_sha256'],
        'validation_original_annotation_sha256':val_split['original_split_sha256'],
        'training_matches':train_split['original_matches'], 'validation_matches':val_split['original_matches'],
        'training_videos':[r['sample']['video'] for r in train_rows],
        'validation_videos':[r['sample']['video'] for r in val_rows],
        'training_clips':len(train_rows), 'validation_clips':len(val_rows), 'seed':args.seed,
        'training_schedule':{'max_epochs':args.epochs, 'patience':args.patience, 'learning_rate':args.learning_rate,
                             'batch_size':args.batch_size, 'clip_len':128, 'optimizer':'AdamW', 'weight_decay':.01},
        'best_validation':best, 'threshold_selection':'shared grid on original val only, maximum micro-F1 at 0.5 seconds',
        'target_derivation':{'kind':'Gaussian training-only smoothing of original exact event frames', 'sigma_sec':.08, 'truncate_sigma':3},
        'new_annotations_created':False, 'test_labels_used_for_training_or_selection':False,
        'probabilities_calibrated':False, 'production_default_approved':False,
        'code_hashes':{'trainer':file_hash(Path(__file__)), 'adapter':file_hash(ROOT/'src/volleymole/action_spotting.py')},
        'limitations':['Small original training subset and one validation match; no claim to reproduce STES paper.',
                      'Source videos have no audio; no proof of a touch, outcome, humor, or highlight worth.']}
    save_json(args.output.with_suffix('.manifest.json'), manifest)
    save_json(args.report, {'manifest':manifest, 'history':history, 'elapsed_sec':time.monotonic()-began})
    print(json.dumps({'checkpoint':str(args.output), 'best_validation':best, 'elapsed_sec':time.monotonic()-began}), flush=True)


def evaluate(args):
    from volleymole.action_spotting import LocalActionSpotter
    began = time.monotonic()
    split = load_split(args.manifest, args.split)
    model_manifest = json.loads(args.checkpoint.with_suffix('.manifest.json').read_text())
    if args.split == 'test':
        seen = set(model_manifest['training_matches']+model_manifest['validation_matches'])
        if seen & set(split['original_matches']):
            raise ValueError('Test match was used for training or selection')
        if args.report.is_file():
            raise ValueError('Sealed test report already exists; do not repeatedly tune on held-out results')
        if args.exclude_manifest:
            old = json.loads(args.exclude_manifest.read_text())
            if {s['video'] for s in split['samples']} & {s['video'] for s in old['samples']}:
                raise ValueError('Sealed test selection overlaps previously inspected diagnostic examples')
    adapter = LocalActionSpotter(args.checkpoint, args.backbone, args.device)
    rows = features_for(split, adapter.backbone, args.cache, args.feature_batch)
    probabilities = [dense_probabilities(adapter.head, row['features'], adapter.config['window'], adapter.config['stride']) for row in rows]
    scores = {}
    for tolerance in (.2, .5, 1.):
        scores[str(tolerance)], predictions = score_rows(rows, probabilities, adapter.config['threshold'], adapter.config['nms_sec'], tolerance)
    report = {'benchmark':'VNL-STES local action proposal spotting', 'split':args.split, 'metrics':scores,
        'complete':True, 'expected_clips':len(rows), 'completed_clips':len(rows), 'failures':[],
        'matches':split['original_matches'], 'manifest_sha256':split['manifest_sha256'],
        'original_annotation_sha256':split['original_split_sha256'], 'model_sha256':adapter.sha256,
        'inference':adapter.config, 'configuration_frozen_at_utc':model_manifest['created_at_utc'],
        'new_annotations_created':False, 'test_labels_used_for_training_or_selection':False,
        'predictions':predictions, 'elapsed_sec':time.monotonic()-began,
        'limitations':model_manifest['limitations']+['Held-out results cover only the explicitly listed original clips.']}
    save_json(args.report, report)
    print(json.dumps({'complete':True, 'metrics':scores, 'elapsed_sec':time.monotonic()-began}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--backbone', type=Path, default=Path('models/actions/resnet18-f37072fd.pth'))
    parser.add_argument('--cache', type=Path, default=Path('runs/action_accuracy/features'))
    parser.add_argument('--feature-batch', type=int, default=64)
    commands = parser.add_subparsers(dest='command', required=True)
    train_parser = commands.add_parser('train')
    train_parser.add_argument('--train-manifest', type=Path, required=True)
    train_parser.add_argument('--val-manifest', type=Path, required=True)
    train_parser.add_argument('--output', type=Path, default=Path('models/actions/vnl_resnet18_gru.pt'))
    train_parser.add_argument('--report', type=Path, default=Path('runs/action_accuracy/local_training.json'))
    train_parser.add_argument('--epochs', type=int, default=60)
    train_parser.add_argument('--patience', type=int, default=12)
    train_parser.add_argument('--batch-size', type=int, default=8)
    train_parser.add_argument('--learning-rate', type=float, default=.001)
    train_parser.add_argument('--seed', type=int, default=20260912)
    eval_parser = commands.add_parser('evaluate')
    eval_parser.add_argument('--manifest', type=Path, required=True)
    eval_parser.add_argument('--split', choices=('val', 'test'), required=True)
    eval_parser.add_argument('--checkpoint', type=Path, default=Path('models/actions/vnl_resnet18_gru.pt'))
    eval_parser.add_argument('--report', type=Path, required=True)
    eval_parser.add_argument('--exclude-manifest', type=Path)
    args = parser.parse_args()
    if args.feature_batch <= 0:
        parser.error('Feature batch must be positive')
    if args.command == 'train' and (args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('Training schedule must be finite and positive')
    torch.set_num_threads(4); cv2.setNumThreads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
