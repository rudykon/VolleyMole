"""Shared decode: each PTS frame feeds state/actions/person/VballNet and optional OCR.

The auxiliary YOLO ball model only sees frames where VballNet has no detection
and the action detector has no ball evidence. No interpolation is emitted as a
neural detection. Independent inference remains available for parity checks.
"""
import csv
import json

from .common import save_json
from .detectors import Detector
from .state_model import StateClassifier
from .tracker import BallTracker
from .jersey import JerseyReader
from .video import chunks, decode


def fallback_indices(ball_rows, actions):
    if len(ball_rows) != len(actions):
        raise ValueError('Action and VballNet batch lengths disagree')
    return [i for i,(ball,raw) in enumerate(zip(ball_rows,actions))
            if not ball['Visibility'] and not any(a['class']=='ball' and a['confidence']>=.25 for a in raw)]


def shared_inference(args, registry, device):
    for kind in ('analytics','tracking','player'):
        (args.output/kind).mkdir(parents=True, exist_ok=True)
    state_model = StateClassifier(registry,device)
    action = Detector(registry,'action',device,half=args.half)
    person = Detector(registry,'person',device,half=args.half)
    auxiliary = Detector(registry,'ball',device,half=args.half)
    tracker = BallTracker(registry,device,args.output/'tracking/profiles')
    reader = JerseyReader(registry,args.number,device,args.confidence,
        runtime_directory=args.output/'player/ocr_runtime') if args.number is not None else None
    windows, count = [], 0
    skipped_visible = skipped_action_ball = 0
    with (args.output/'analytics/detections.jsonl').open('w') as out, \
            (args.output/'tracking/ball.csv').open('w') as csv_out, \
            (args.output/'tracking/source_pts.csv').open('w') as pts:
        writer = csv.DictWriter(csv_out,fieldnames=['Frame','Visibility','X','Y','Radius','Confidence','SourceTime','evidence'])
        writer.writeheader()
        # 90 is the least common multiple of state windows (30) and VballNet (9).
        # Preserve both models' original sequence boundaries, including EOF tails.
        for packets in chunks(decode(args.video,max_frames=args.max_frames),90):
            images = [p.pixels for p in packets]
            balls = [row for batch in chunks(packets,9) for row in tracker.predict(batch)]
            # Preserve the legacy 30-frame detector batch layout for exact parity.
            actions = [row for batch in chunks(images,30) for row in action.detect(batch)]
            people = [row for batch in chunks(images,30) for row in person.detect(batch)]
            missing = fallback_indices(balls,actions)
            extra_rows = auxiliary.detect([images[i] for i in missing])
            extras = dict(zip(missing,extra_rows))
            states = []
            for batch in chunks(packets,30):
                state = state_model.classify([p.pixels for p in batch])
                states.extend([state]*len(batch))
                windows.append({'start_frame':batch[0].index,'end_frame':batch[-1].index,
                    'start_s':batch[0].time_sec,'end_s':batch[-1].time_sec,'state':state.label,
                    'confidence':state.confidence,'probabilities':state.probabilities,
                    'sampled_frames':[batch[i].index for i in state.sampled_indices]})
            for i,(packet,ball,raw_actions,players,state) in enumerate(zip(packets,balls,actions,people,states)):
                action_balls = [a for a in raw_actions if a['class']=='ball' and a['confidence']>=.25]
                if ball['Visibility']:
                    skipped_visible += 1
                    status = 'not_run_vball_visible'
                    evidence_balls = []
                elif i not in extras:
                    skipped_action_ball += 1
                    status = 'not_run_action_ball_available'
                    evidence_balls = [dict(a,origin='action_detector') for a in action_balls]
                else:
                    status = 'ran_vball_missing'
                    evidence_balls = [dict(a,origin='auxiliary_ball_detector') for a in extras[i]]
                row = {**packet.clock(),'state':state.label,'state_confidence':state.confidence,
                    'state_probabilities':state.probabilities,'primary_ball':ball,
                    'ball':max(evidence_balls,key=lambda b:b['confidence']) if evidence_balls else None,
                    'actions':[a for a in raw_actions if a['class'] not in ('ball','serve')],
                    'players':players,'raw_actions':raw_actions,'raw_balls':extras.get(i,[]),
                    'auxiliary_ball_status':status}
                out.write(json.dumps(row,allow_nan=False)+'\n')
                writer.writerow(ball)
                pts.write(f'{packet.source_sec:.9f}\n')
                if reader is not None:
                    reader.consume(packet,players)
                count += 1
            if count % 900 == 0:
                print(f'shared: {count} frames, auxiliary ball on {auxiliary.frames}',flush=True)
    model_names = ['state_weights','state_config','state_processor','action','person','ball','vball']
    if reader is not None:
        model_names += ['ocr_recognizer','ocr_detector']
    summary = {'status':'complete' if args.max_frames is None else 'partial_smoke',
        'input':str(args.video),'processed_frames':count,'device':device,'state_windows':windows,
        'counts':{'decode_passes':1,'decoded_frames':count,'state_calls':state_model.calls,
            'vball_calls':tracker.calls,'vball_frames':tracker.frames,'person_frames':person.frames,
            'action_frames':action.frames,'ball_detector_frames':auxiliary.frames,
            'ball_skipped_vball_visible':skipped_visible,'ball_skipped_action_ball':skipped_action_ball,
            'ocr_calls':reader.calls if reader else 0,'ocr_extra_person_detections':0},
        'backend':tracker.backend,'implementation':'shared-pts90-v1'}
    if count != auxiliary.frames+skipped_visible+skipped_action_ball:
        raise RuntimeError('Auxiliary ball scheduling accounting mismatch')
    save_json(args.output/'summary.json',summary)
    save_json(args.output/'analytics/summary.json',summary)
    save_json(args.output/'tracking/summary.json',summary)
    player = reader.result() if reader else {'status':'not_requested','number':None,'detections':[]}
    if reader and args.max_frames is not None:
        player['status'] = 'partial_smoke'
    save_json(args.output/'player/index.json',player)
    return model_names
