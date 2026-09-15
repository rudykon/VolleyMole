"""Bounded, timestamp-grounded action observations for rapid volleyball play.

Every source interval is covered. Overlapping context supplies the action's
before/after while disjoint ownership avoids counting it twice. Observations
remain model predictions; they are neither new labels nor confirmed outcomes.
"""
import hashlib
import json
import math
import time
from pathlib import Path

from .common import APP, digest, read_json, save_json
from .semantic import request_json, sampled_evidence

CLASSES = ('serve', 'receive', 'set', 'spike', 'block', 'score')
PROMPT = """Observe a short continuous volleyball video with exact frame timestamps.
Find individual visible volleyball actions, not a generic expected rally sequence.
Classes: serve=service contact; receive=serve reception or defensive dig;
set=setting contact; spike=attack hit; block=block attempt at the net;
score=directly visible point-ending event. A raised arm or jump alone does not
establish a contact. Do not infer an unseen reception/set just because a spike
follows. Do not report the same action across multiple adjacent frames.
For each action choose the supplied frame ID nearest its actual occurrence,
and a later supplied frame supporting its immediate aftermath. If that later
frame is unavailable, use after_frame_id=null and aftermath=unknown. It is valid to
return zero actions. Return unknown rather than invent a ball, team, winner,
score, sound or successful save. The clip may begin or end during play.
Give a brief concrete visible observation, not the definition of the class.
JSON only: {"events":[{"label":"serve|receive|set|spike|block|score",
"frame_id":"frame_00000","after_frame_id":"frame_00001",
"confidence":0.0,"observable":true,"aftermath":"continues|point_ends|unknown",
"observation":"brief direct visual evidence"}],"uncertainty":"sampling/visibility limitations"}.
An action with insufficient visual evidence must have observable=false.
"""


def action_windows(duration, fps=8., max_frames=24, overlap_sec=.5):
    if (not math.isfinite(duration) or duration<=0 or not math.isfinite(fps) or fps<=0
            or type(max_frames) is not int or max_frames<4):
        raise ValueError('Positive finite duration/FPS and at least four frames required')
    size=max_frames/fps
    if not math.isfinite(overlap_sec) or not 0<=overlap_sec<size:
        raise ValueError('Overlap must be shorter than the context')
    spans=[];start=0.
    while start<duration-1e-8:
        end=min(duration,start+size)
        spans.append((start,end))
        if end==duration:break
        start+=size-overlap_sec
    boundaries=[0.]+[(spans[i][1]+spans[i+1][0])/2 for i in range(len(spans)-1)]+[duration]
    return [{'index':i,'start':a,'end':b,'own_start':boundaries[i],'own_end':boundaries[i+1]}
            for i,(a,b) in enumerate(spans)]


def grounded_predictions(data, evidence, job):
    if (not isinstance(data,dict) or set(data)!={'events','uncertainty'}
            or not isinstance(data['events'],list) or len(data['events'])>48
            or not isinstance(data['uncertainty'],str)):
        raise ValueError('Invalid grounded action response')
    refs={r['id']:r['start_sec'] for r in evidence if r['kind']=='frame'}
    accepted=[];rejected=[]
    required={'label','frame_id','after_frame_id','confidence','observable','aftermath','observation'}
    for row in data['events']:
        if (not isinstance(row,dict) or set(row)!=required or row['label'] not in CLASSES
                or type(row['observable']) is not bool or row['aftermath'] not in ('continues','point_ends','unknown')
                or not isinstance(row['observation'],str) or not row['observation'].strip()
                or len(row['observation'])>600):
            raise ValueError('Invalid grounded action fields')
        if (type(row['confidence']) not in (int,float) or not math.isfinite(row['confidence'])
                or not 0<=row['confidence']<=1):raise ValueError('Invalid action confidence')
        if not isinstance(row['frame_id'],str) or row['frame_id'] not in refs:
            raise ValueError('Action must point to supplied frame IDs')
        when=refs[row['frame_id']]
        if row['after_frame_id'] is None:
            if row['aftermath']!='unknown':raise ValueError('Missing later frame requires unknown aftermath')
        elif (not isinstance(row['after_frame_id'],str) or row['after_frame_id'] not in refs
                or refs[row['after_frame_id']]<=when):
            raise ValueError('Aftermath must reference a later observed frame')
        if not row['observable']:
            rejected.append({'reason':'unobservable','prediction':row});continue
        if not job['own_start']<=when<job['own_end']:
            rejected.append({'reason':'owned_by_neighbor_context','prediction':row});continue
        accepted.append({'label':row['label'],'time_sec':when,'confidence':row['confidence'],
            'evidence_ids':[row['frame_id']]+([row['after_frame_id']] if row['after_frame_id'] is not None else []),
            'observation':row['observation'],'aftermath':row['aftermath'],
            'time_basis':'actual supplied video frame PTS','confirmed_contact':None,
            'outcome_verified':False})
    # Exactly repeated frame/class claims in one response are one observation.
    unique={}
    for row in accepted:
        key=(row['label'],row['time_sec'])
        if key not in unique or row['confidence']>unique[key]['confidence']:unique[key]=row
    return sorted(unique.values(),key=lambda r:(r['time_sec'],r['label'])),rejected


def video_frame_content(content,evidence,source):
    """vLLM documented video/jpeg transport with explicit original frame indices.

The exact client sample is retained; server-side sampling is requested off.
Acceptance alone is not proof that a third-party server honors these fields.
"""
    from fractions import Fraction
    fps=float(Fraction(source['nominal_fps']))
    frames=[c['image_url']['url'].split(',',1)[1] for c in content if c['type']=='image_url']
    refs=[r for r in evidence if r['kind']=='frame']
    if len(frames)!=len(refs):raise ValueError('Video transport frame count mismatch')
    metadata={'fps':fps,'frames_indices':[round(r['start_sec']*fps) for r in refs],
              'total_num_frames':source['frame_count'],'duration':source['duration_sec'],
              'do_sample_frames':False}
    parts=[{'type':'video_url','video_url':{'url':'data:video/jpeg;base64,'+','.join(frames)}}]
    return parts,{'media_io_kwargs':{'video':metadata},'mm_processor_kwargs':{'do_sample_frames':False}}


def observe_actions(source,job,cache,settings,deadline):
    width=min(settings.get('width',398),source['width'])
    fps=settings.get('fps',8.)
    transport=settings.get('transport','frames')
    if transport not in ('frames','video_frames'):raise ValueError('Unsupported action transport')
    identity={'version':1,'source_sha256':digest(source['path']),'job':job,'fps':fps,'width':width,
        'transport':transport,'model':settings['model'],'endpoint':settings['endpoint'],
        'max_tokens':settings.get('max_tokens',1024),
        'prompt':PROMPT,'module_sha256':digest(APP/'temporal_understanding.py'),
        'sampler_sha256':digest(APP/'semantic.py')}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=Path(cache)/f'{signature}.json'
    if path.is_file():
        saved=read_json(path)
        if saved.get('signature')==signature:
            predicted,rejected=grounded_predictions(saved['raw'],saved['evidence'],job)
            return {**saved,'events':predicted,'rejected':rejected,'cached':True}
    began=time.monotonic()
    evidence,content,_=sampled_evidence(source,job['start'],job['end'],fps,width,deadline=deadline)
    extra={}
    if transport=='video_frames':content,extra=video_frame_content(content,evidence,source)
    content.insert(0,{'type':'text','text':json.dumps({'context':[job['start'],job['end']],
        'report_interval':[job['own_start'],job['own_end']],'audio_available':False,
        'frames':[{k:r[k] for k in ('id','start_sec')} for r in evidence]})})
    payload={'model':settings['model'],'temperature':0,'max_tokens':settings.get('max_tokens',1024),
        'messages':[{'role':'system','content':PROMPT},{'role':'user','content':content}],
        'response_format':{'type':'json_object'},**extra}
    remaining=min(settings.get('timeout',90),deadline-time.monotonic())
    if remaining<=0:raise TimeoutError('Action observation deadline')
    data,metadata=request_json(settings['endpoint'],settings['key'],payload,remaining)
    predicted,rejected=grounded_predictions(data,evidence,job)
    result={'signature':signature,'identity':identity,'job':job,'events':predicted,'rejected':rejected,
        'raw':data,'evidence':evidence,'request':metadata,'elapsed_sec':time.monotonic()-began,'cached':False,
        'requested_fps':fps,'sample_times_sec':[r['start_sec'] for r in evidence],
        'server_sampling_verified':False if transport=='video_frames' else None}
    save_json(path,result)
    return result
