#!/usr/bin/env python3
"""Export bounded local action proposals using actual decoded video PTS.

Only the verified 25 FPS domain is supported. RGB frames are resized as they
arrive; disk-backed features and overlapping GRU windows avoid retaining a
whole match of decoded images. A deadline produces an explicit partial prefix.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volleymole.action_spotting import CLASSES, FEATURE_DIM, LocalActionSpotter, file_hash, proposals
from volleymole.common import probe, save_json
from volleymole.video import decode


def verified_rate(source):
    try:
        return all(Fraction(source[key]) == 25 for key in ('nominal_fps', 'average_fps'))
    except (ValueError, ZeroDivisionError, KeyError):
        return False


def export(args):
    began = time.monotonic(); deadline = began+args.total_timeout
    source = probe(args.video)
    source_hash = file_hash(args.video)
    checkpoint_hash = file_hash(args.checkpoint)
    model_manifest_path = args.checkpoint.with_suffix('.manifest.json')
    model_manifest = json.loads(model_manifest_path.read_text())
    if model_manifest['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Frozen action checkpoint changed')
    result = {'schema_version':1,
        'source':{'sha256':source_hash, 'duration_sec':source['duration_sec'],
                  'container_start_sec':source['start_sec'], 'time_basis':'source_relative_to_container_start',
                  'nominal_fps':source['nominal_fps'], 'average_fps':source['average_fps']},
        'model':{'sha256':checkpoint_hash, 'manifest_sha256':file_hash(model_manifest_path),
                 'backbone_sha256':model_manifest['backbone_sha256']},
        'status':'partial', 'events':[], 'covered_start_sec':None, 'covered_end_sec':None,
        'processed_frames':0, 'finalized_frames':0, 'sampling_fps':25,
        'uncertainty':'Local visual predictions are hypotheses, not confirmed contacts, successes or points.',
        'parameters':{'total_timeout_sec':args.total_timeout, 'max_cache_mib':args.max_cache_mib,
                      'feature_batch':args.feature_batch},
        'new_annotations_created':False}

    def finish(status, reason):
        result.update(status=status, reason=reason, elapsed_sec=time.monotonic()-began)
        save_json(args.output, result)
        return result

    if not verified_rate(source):
        return finish('unsupported', 'The trained action model requires verified 25 FPS source video.')
    if time.monotonic() >= deadline:
        return finish('partial', 'Time budget exhausted during source verification.')
    adapter = LocalActionSpotter(args.checkpoint, args.backbone, args.device)
    result['model']['feature_compute_policy'] = adapter.backbone.compute_policy
    window, stride = adapter.config['window'], adapter.config['stride']
    if not 0 < stride <= window or adapter.config['fps'] != 25:
        raise ValueError('Unsupported frozen temporal inference configuration')
    # Per-frame storage: float16 features, float64 time, int64 PTS, two int32
    # time-base values, float32 six-class logit sum, uint16 overlap count.
    bytes_per_frame = FEATURE_DIM*2+8+8+8+len(CLASSES)*4+2
    capacity = min(max(source['frame_count'], int(math.ceil(source['duration_sec']*25))+4),
                   int(args.max_cache_mib*1024**2//bytes_per_frame))
    if capacity < min(window, max(1, source['frame_count'] or int(math.ceil(source['duration_sec']*25)))):
        return finish('partial', 'Feature cache budget is smaller than one complete temporal window.')
    args.cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='action-evidence-', dir=args.cache) as directory:
        directory = Path(directory)
        def mapped(name, dtype, shape):
            return np.memmap(directory/name, mode='w+', dtype=dtype, shape=shape)
        features = mapped('features.bin', np.float16, (capacity, FEATURE_DIM))
        times = mapped('times.bin', np.float64, (capacity,))
        pts = mapped('pts.bin', np.int64, (capacity,))
        bases = mapped('bases.bin', np.int32, (capacity, 2))
        sums = mapped('sum.bin', np.float32, (capacity, len(CLASSES))); sums[:] = 0
        counts = mapped('counts.bin', np.uint16, (capacity,)); counts[:] = 0
        encoded, next_start, last_end = 0, 0, 0
        pending, pending_clocks = [], []
        previous_time = None
        stopped = None
        completed_decode = False

        @torch.inference_mode()
        def score_window(start, end):
            nonlocal next_start, last_end
            tensor = torch.as_tensor(np.asarray(features[start:end], dtype=np.float32), device=args.device)[None]
            values = adapter.head(tensor).squeeze(0).cpu().numpy()
            sums[start:end] += values; counts[start:end] += 1
            next_start = start+stride; last_end = end

        def encode_pending():
            nonlocal encoded, stopped
            if not pending:
                return
            if time.monotonic() >= deadline:
                stopped = 'Time budget exhausted before the next CNN batch.'
                return
            values = adapter.backbone.encode(pending, args.feature_batch)
            end = encoded+len(values)
            features[encoded:end] = values
            for i, clock in enumerate(pending_clocks, encoded):
                times[i], pts[i], bases[i] = clock
            encoded = end; pending.clear(); pending_clocks.clear()
            while next_start+window <= encoded:
                if time.monotonic() >= deadline:
                    stopped = 'Time budget exhausted before the next temporal window.'
                    return
                score_window(next_start, next_start+window)

        iterator = decode(args.video)
        try:
            for packet in iterator:
                if time.monotonic() >= deadline:
                    stopped = 'Time budget exhausted while decoding source evidence.'
                    break
                when = packet.time_sec
                if (not math.isfinite(when) or when < 0 or when >= source['duration_sec']
                        or (previous_time is not None and not math.isclose(when-previous_time, .04, abs_tol=.0002))):
                    result['processed_frames'] = encoded
                    return finish('unsupported', 'Actual decoded PTS do not follow the verified 25 FPS domain.')
                previous_time = when
                if encoded+len(pending) >= capacity:
                    stopped = 'Feature cache byte budget exhausted before source video ended.'
                    break
                # Retain only one source-resolution frame and a bounded low-res batch.
                pending.append(cv2.resize(packet.pixels, (398, 224)))
                pending_clocks.append((when, packet.pts, [packet.time_base.numerator, packet.time_base.denominator]))
                if len(pending) == args.feature_batch:
                    encode_pending()
                    if stopped:
                        break
            else:
                completed_decode = True
        except ValueError as exc:
            if 'presentation timestamp' in str(exc):
                result['processed_frames'] = encoded
                return finish('unsupported', 'Decoded source evidence lacks a valid monotonic PTS clock.')
            raise
        finally:
            iterator.close()
        if not stopped:
            encode_pending()
        if completed_decode and not stopped:
            if last_end < encoded:
                if time.monotonic() >= deadline:
                    stopped = 'Time budget exhausted before final temporal context.'
                else:
                    score_window(next_start, encoded)
            finalized = encoded if not stopped else min(next_start, encoded)
        else:
            # Future overlapping windows could change this tail. Export only
            # the prefix whose full trained context has already been observed.
            finalized = min(next_start, encoded)
        result.update(processed_frames=encoded, finalized_frames=finalized,
                      estimated_cache_bytes=capacity*bytes_per_frame)
        if finalized and np.all(counts[:finalized] > 0):
            logits = np.asarray(sums[:finalized], dtype=np.float64)/np.asarray(counts[:finalized])[:, None]
            probabilities = (1/(1+np.exp(-np.clip(logits, -50, 50)))).astype(np.float32)
            rows = proposals(probabilities, times[:finalized], adapter.config['threshold'], adapter.config['nms_sec'])
            result['events'] = [{**row, 'observation_status':'candidate',
                'source_time_sec':row['time_sec'], 'source_frame_index':row['frame_index'],
                'source_pts':int(pts[row['frame_index']]), 'source_time_base':bases[row['frame_index']].tolist(),
                'model_sha256':checkpoint_hash,
                'uncertainty':'Uncalibrated visual proposal; verify the original video before describing a contact or outcome.'}
                for row in rows]
            result['covered_start_sec'] = float(times[0])
            result['covered_end_sec'] = min(source['duration_sec'], float(times[finalized-1])+.04)
        return finish('complete' if completed_decode and not stopped else 'partial',
                      stopped or 'All decoded source frames processed at actual source PTS.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/actions/vnl_resnet18_gru.pt'))
    parser.add_argument('--backbone', type=Path, default=Path('models/actions/resnet18-f37072fd.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('runs/action_accuracy/export_cache'))
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--total-timeout', type=float, default=300)
    parser.add_argument('--max-cache-mib', type=float, default=2048)
    parser.add_argument('--feature-batch', type=int, default=64)
    args = parser.parse_args()
    if (any(not math.isfinite(v) or v <= 0 for v in (args.total_timeout, args.max_cache_mib))
            or not 1 <= args.feature_batch <= 128):
        parser.error('Finite positive budgets and a bounded feature batch are required')
    torch.set_num_threads(4); cv2.setNumThreads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    result = export(args)
    print(json.dumps({key:result[key] for key in ('status', 'processed_frames', 'finalized_frames', 'reason', 'elapsed_sec')}, ensure_ascii=False))


if __name__ == '__main__': main()
