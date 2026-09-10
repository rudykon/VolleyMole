"""Standalone inference modules; independent decode passes form the phase-1 gate."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

from .common import identity, save_json
from .models import ModelRegistry
from .video import chunks, decode
from .detectors import Detector, resolve_device
from .telemetry import UsageMonitor
from .gpu_stages import parse_devices


def analytics(args, registry, device):
    from .state_model import StateClassifier
    state_model = StateClassifier(registry, device)
    action = Detector(registry, 'action', device, half=args.half)
    ball = Detector(registry, 'ball', device, half=args.half)
    person = Detector(registry, 'person', device, half=args.half)
    windows, count = [], 0
    with (args.output/'detections.jsonl').open('w') as out:
        for packets in chunks(decode(args.video, max_frames=args.max_frames), 30):
            images = [p.pixels for p in packets]
            state = state_model.classify(images)
            actions, balls, people = action.detect(images), ball.detect(images), person.detect(images)
            window = {'start_frame': packets[0].index, 'end_frame': packets[-1].index,
                      'start_s': packets[0].time_sec, 'end_s': packets[-1].time_sec,
                      'state': state.label, 'confidence': state.confidence,
                      'probabilities': state.probabilities,
                      'sampled_frames': [packets[i].index for i in state.sampled_indices]}
            windows.append(window)
            for packet, raw_actions, raw_balls, players in zip(packets, actions, balls, people):
                row = {**packet.clock(), 'state': state.label, 'state_confidence': state.confidence,
                       'state_probabilities': state.probabilities,
                       'ball': max(raw_balls,key=lambda b:b['confidence']) if raw_balls else None,
                       'actions': [a for a in raw_actions if a['class'] not in ('ball','serve')],
                       'players': players, 'raw_actions': raw_actions, 'raw_balls': raw_balls}
                out.write(json.dumps(row, allow_nan=False)+'\n')
                count += 1
            if count % 900 == 0:
                print(f'analytics: {count} frames', flush=True)
    save_json(args.output/'summary.json', {'status': 'complete' if args.max_frames is None else 'partial_smoke',
        'input': str(args.video), 'processed_frames': count, 'state_windows': windows, 'device': device,
        'counts': {'decode_passes': 1, 'decoded_frames': count, 'state_calls': state_model.calls,
                   'action_frames': action.frames, 'ball_detector_frames': ball.frames, 'person_frames': person.frames}})
    return ['state_weights', 'state_config', 'state_processor', 'action', 'ball', 'person']


def tracking(args, registry, device):
    from .tracker import BallTracker
    tracker = BallTracker(registry, device, args.output/'profiles')
    with (args.output/'ball.csv').open('w') as out, (args.output/'source_pts.csv').open('w') as pts:
        writer = csv.DictWriter(out, fieldnames=['Frame','Visibility','X','Y','Radius','Confidence','SourceTime','evidence'])
        writer.writeheader()
        for packets in chunks(decode(args.video, max_frames=args.max_frames), tracker.sequence_length):
            for row in tracker.predict(packets):
                writer.writerow(row)
                pts.write(f'{row["SourceTime"]:.9f}\n')
            if tracker.frames % 900 == 0:
                print(f'tracking: {tracker.frames} frames', flush=True)
    save_json(args.output/'summary.json', {'status': 'complete' if args.max_frames is None else 'partial_smoke',
        'input': str(args.video), 'processed_frames': tracker.frames, 'backend': tracker.backend,
        'counts': {'decode_passes':1, 'decoded_frames':tracker.frames, 'vball_calls':tracker.calls}})
    return ['vball']


def player(args, registry, device):
    from .jersey import JerseyReader
    if args.number is None:
        save_json(args.output/'index.json', {'status':'not_requested','number':None,'detections':[]})
        return []
    reader = JerseyReader(registry, args.number, device, args.confidence,
                          runtime_directory=args.output/'ocr_runtime')
    person = Detector(registry, 'person', device, half=args.half)
    count = 0
    for packet in decode(args.video, max_frames=args.max_frames):
        count += 1
        if packet.time_sec+1e-6 >= reader.next_sample:
            people = person.detect([packet.pixels])[0]
            reader.consume(packet, people)
    result = reader.result()
    if args.max_frames is not None:
        result['status'] = 'partial_smoke'
    result['counts'] = {'decode_passes':1, 'decoded_frames':count, 'person_frames':person.frames}
    save_json(args.output/'index.json', result)
    return ['person', 'ocr_recognizer', 'ocr_detector']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['analytics','tracking','player','shared'], required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--devices', type=parse_devices,
                        help='四卡共享推理：cuda:0,cuda:1,cuda:2,cuda:3（状态/动作/人物/球轨迹）')
    parser.add_argument('--number', type=int)
    parser.add_argument('--confidence', type=float, default=.75)
    parser.add_argument('--max-frames', type=int, help='smoke test only; never accepted as a full-match cache')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--pipeline-depth', type=int, choices=range(1,5), default=1)
    parser.add_argument('--auxiliary-device')
    parser.add_argument('--vball-engine', choices=['ort','ort-bound'], default='ort')
    args = parser.parse_args(argv)
    if args.kind != 'shared' and (args.pipeline_depth != 1 or args.auxiliary_device or args.vball_engine != 'ort'):
        parser.error('Performance options require --kind shared')
    if args.auxiliary_device and (not args.devices or args.auxiliary_device not in args.devices):
        parser.error('--auxiliary-device must be one of --devices')
    if args.devices and (args.kind != 'shared' or args.device != 'auto'):
        parser.error('--devices requires --kind shared and cannot be combined with --device')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    args.video = args.video.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(args.output/'yolo_config'))
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    import torch
    torch.set_num_threads(4)
    if args.devices:
        args.devices = [resolve_device(d) for d in args.devices]
    device = args.devices[0] if args.devices else resolve_device(args.device)
    registry = ModelRegistry(args.models)
    from .shared import shared_inference
    with UsageMonitor(args.devices or device) as usage:
        models = {'analytics':analytics, 'tracking':tracking, 'player':player,
                  'shared':shared_inference}[args.kind](args,registry,device)
    save_json(args.output/'telemetry.json', usage.report())
    save_json(args.output/'provenance.json', {'project':'VolleyMole', 'mode':'package_fresh_inference',
        'source':identity(args.video), 'models':registry.verify(models) if models else {},
        'parameters':{'device':device, 'devices':args.devices, 'half':args.half, 'max_frames':args.max_frames,
                      'pipeline_depth':args.pipeline_depth,'auxiliary_device':args.auxiliary_device,'vball_engine':args.vball_engine},
        'time_basis':'source PTS minus container start',
        'implementation': ('shared-pts90-four-gpu-v1' if args.devices else 'shared-pts90-v1')
                         if args.kind=='shared' else 'independent-decode-phase1'})
    print(f'{args.kind}: {usage.elapsed:.3f}s on {args.devices or device}', flush=True)


if __name__ == '__main__':
    main()
