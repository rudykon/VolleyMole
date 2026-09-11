"""Shared decode: each PTS frame feeds state/actions/person/VballNet and optional OCR.

The auxiliary YOLO ball model only sees frames where VballNet has no detection
and the action detector has no ball evidence. No interpolation is emitted as a
neural detection. Independent inference remains available for parity checks.
"""
import csv
import json
from contextlib import nullcontext
from time import perf_counter

from .common import save_json
from .detectors import Detector
from .state_model import StateClassifier
from .tracker import BallTracker
from .jersey import JerseyReader
from .video import chunks, decode
from .gpu_stages import GPUStages, ROLES
from .performance import Timings
from .pipeline import prefetch_batches, ordered_inference


def submit_batch(packets, state_model, action, person, auxiliary, tracker, workers, timings):
    images = [p.pixels for p in packets]
    state = workers.submit('state', classify_batch, state_model, packets)
    ball = workers.submit('tracking', track_batch, tracker, packets)
    actions = workers.submit('action', detect_batch, action, images)
    people = workers.submit('person', detect_batch, person, images)
    def supplement():
        with timings.measure('auxiliary_dependency_wait'):
            missing = fallback_indices(ball.result(), actions.result())
        return dict(zip(missing, auxiliary.detect([images[i] for i in missing])))
    extra = workers.submit('auxiliary', supplement)
    return packets, ball, actions, people, extra, state


def resolve_batch(job):
    packets, ball, action, person, extra, state = job
    balls, actions, people, extras = ball.result(), action.result(), person.result(), extra.result()
    states, windows = state.result()
    if any(len(rows) != len(packets) for rows in (balls, actions, people, states)):
        raise RuntimeError('Shared inference returned an incomplete batch')
    if len(extras) != len(fallback_indices(balls, actions)):
        raise RuntimeError('Shared inference returned an incomplete auxiliary batch')
    return balls, actions, people, extras, states, windows


def fallback_indices(ball_rows, actions):
    if len(ball_rows) != len(actions):
        raise ValueError('Action and VballNet batch lengths disagree')
    return [i for i,(ball,raw) in enumerate(zip(ball_rows,actions))
            if not ball['Visibility'] and not any(a['class']=='ball' and a['confidence']>=.25 for a in raw)]


def track_batch(tracker, packets):
    return [row for batch in chunks(packets,9) for row in tracker.predict(batch)]


def detect_batch(detector, images):
    # Retain 30-frame outer / 8-frame inner batches, also in multi-GPU mode.
    return [row for batch in chunks(images,30) for row in detector.detect(batch)]


def classify_batch(model, packets):
    states, windows = [], []
    for batch in chunks(packets,30):
        state = model.classify([p.pixels for p in batch])
        states.extend([state]*len(batch))
        windows.append({'start_frame':batch[0].index,'end_frame':batch[-1].index,
            'start_s':batch[0].time_sec,'end_s':batch[-1].time_sec,'state':state.label,
            'confidence':state.confidence,'probabilities':state.probabilities,
            'sampled_frames':[batch[i].index for i in state.sampled_indices]})
    return states, windows


def infer_batch(packets, state_model, action, person, auxiliary, tracker, workers=None):
    images = [p.pixels for p in packets]
    if workers is None:
        balls = track_batch(tracker, packets)
        actions = detect_batch(action, images)
        people = detect_batch(person, images)
        missing = fallback_indices(balls, actions)
        extra_rows = auxiliary.detect([images[i] for i in missing])
        states, windows = classify_batch(state_model, packets)
    else:
        state_future = workers.submit('state', classify_batch, state_model, packets)
        ball_future = workers.submit('tracking', track_batch, tracker, packets)
        action_future = workers.submit('action', detect_batch, action, images)
        person_future = workers.submit('person', detect_batch, person, images)
        balls, actions = ball_future.result(), action_future.result()
        missing = fallback_indices(balls, actions)
        # Same tracking worker/device: never run auxiliary or tracker twice on
        # one model concurrently, and do not speculate on the fallback mask.
        extra_future = workers.submit('tracking', auxiliary.detect, [images[i] for i in missing])
        people = person_future.result()
        states, windows = state_future.result()
        extra_rows = extra_future.result()
    if any(len(rows) != len(packets) for rows in (balls, actions, people, states)) or len(extra_rows) != len(missing):
        raise RuntimeError('Shared inference returned an incomplete batch')
    return balls, actions, people, dict(zip(missing, extra_rows)), states, windows


def shared_inference(args, registry, device):
    timings = Timings()
    load_started = perf_counter()
    for kind in ('analytics','tracking','player'):
        (args.output/kind).mkdir(parents=True, exist_ok=True)
    devices = getattr(args, 'devices', None)
    assigned = dict(zip(ROLES, devices)) if devices else dict.fromkeys(ROLES, device)
    auxiliary_device = getattr(args, 'auxiliary_device', None) or assigned['tracking']
    state_model = StateClassifier(registry,assigned['state'])
    action = Detector(registry,'action',assigned['action'],half=args.half)
    person = Detector(registry,'person',assigned['person'],half=args.half)
    auxiliary = Detector(registry,'ball',auxiliary_device,half=args.half)
    tracker = BallTracker(registry,assigned['tracking'],args.output/'tracking/profiles',
                          engine=getattr(args, 'vball_engine', 'ort'))
    reader = JerseyReader(registry,args.number,assigned['person'],args.confidence,
        runtime_directory=args.output/'player/ocr_runtime') if args.number is not None else None
    windows, count = [], 0
    skipped_visible = skipped_action_ball = 0
    timings.add('model_load', perf_counter()-load_started)
    depth = getattr(args, 'pipeline_depth', 1)
    packets = decode(args.video,max_frames=args.max_frames,timings=timings)
    if getattr(args,'event_frame_cache',None):
        from .event_frames import tap
        packets = tap(packets,args.event_frame_cache,args.event_frame_fps)
    batches = chunks(packets,90)
    with (GPUStages(devices, auxiliary_device=auxiliary_device) if devices else nullcontext()) as workers, \
            (prefetch_batches(batches,timings) if depth > 1 else nullcontext(batches)) as batch_source, \
            (args.output/'analytics/detections.jsonl').open('w') as out, \
            (args.output/'tracking/ball.csv').open('w') as csv_out, \
            (args.output/'tracking/source_pts.csv').open('w') as pts:
        writer = csv.DictWriter(csv_out,fieldnames=['Frame','Visibility','X','Y','Radius','Confidence','SourceTime','evidence'])
        writer.writeheader()
        # 90 is the least common multiple of state windows (30) and VballNet (9).
        # Preserve both models' original sequence boundaries, including EOF tails.
        if workers and depth > 1:
            results = ordered_inference(batch_source,
                lambda p: submit_batch(p,state_model,action,person,auxiliary,tracker,workers,timings),
                resolve_batch,depth,timings)
        else:
            def serial_results():
                for packets in batch_source:
                    with timings.measure('inference_consumer_wait'):
                        if workers and auxiliary_device != assigned['tracking']:
                            result = resolve_batch(submit_batch(packets,state_model,action,person,auxiliary,tracker,workers,timings))
                        else:
                            result = infer_batch(packets,state_model,action,person,auxiliary,tracker,workers)
                    yield packets,result
            results = serial_results()
        for packets,result in results:
            balls, actions, people, extras, states, batch_windows = result
            write_started = perf_counter()
            windows.extend(batch_windows)
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
            timings.add('result_conversion_write_and_ocr', perf_counter()-write_started)
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
    if devices:
        actual = {'state': str(next(state_model.model.parameters()).device),
                  'action': str(action.model.predictor.device),
                  'person': str(person.model.predictor.device),
                  'tracking': tracker.backend['requested']}
        auxiliary_device = str(auxiliary.model.predictor.device) if auxiliary.frames else None
        if actual != assigned or (auxiliary_device is not None and auxiliary_device != (getattr(args,'auxiliary_device',None) or assigned['tracking'])):
            raise RuntimeError('Actual inference model devices disagree with four-GPU assignment')
        summary.update(devices=devices,implementation='shared-pts90-four-gpu-v1',
            gpu_execution={**workers.report(),'actual_model_devices':actual,
                           'auxiliary_actual_device':auxiliary_device})
    if count != auxiliary.frames+skipped_visible+skipped_action_ball:
        raise RuntimeError('Auxiliary ball scheduling accounting mismatch')
    summary['performance'] = {'pipeline_depth':depth,'host':timings.report(),
        'models':{name:model.timings.report() for name,model in
                  [('state',state_model),('action',action),('person',person),('auxiliary',auxiliary),('tracking',tracker)]}}
    save_json(args.output/'summary.json',summary)
    save_json(args.output/'analytics/summary.json',summary)
    save_json(args.output/'tracking/summary.json',summary)
    player = reader.result() if reader else {'status':'not_requested','number':None,'detections':[]}
    if reader and args.max_frames is not None:
        player['status'] = 'partial_smoke'
    save_json(args.output/'player/index.json',player)
    return model_names
