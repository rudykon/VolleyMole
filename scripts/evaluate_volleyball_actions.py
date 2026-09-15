#!/usr/bin/env python3
"""Evaluate VLM action spotting on unchanged, author-labelled VNL-STES test clips.

Predictions are separate artifacts, never dataset annotations. This diagnostic
does not evaluate highlight selection, humor, spatial spotting or edit safety.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from fractions import Fraction
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from volleymole.common import digest, probe, read_json, save_json
from volleymole.semantic import bounded_map, request_json, sampled_evidence

CLASSES = ('serve', 'receive', 'set', 'spike', 'block', 'score')
PROMPT = """You are spotting volleyball actions in a complete silent broadcast clip.
The video is an ordered sequence of frames with explicit source_sec timestamps.
Report only directly supported actions, with their estimated occurrence time in
seconds on this clip's clock. Ontology: serve = initiating service contact;
receive = reception or defensive dig; set = setting for attack; spike = attack
hit; block = blocking contact or attempt at the net; score = observable end of
point. Report each distinct action once. Do not guess a scoring team, sound,
or an occluded contact. Use uncertainty if sampling cannot resolve an action.
Return JSON {"events":[{"label":"serve|receive|set|spike|block|score",
"time_sec":0.0,"evidence_ids":["frame_00000"],"confidence":0.0}],
"uncertainty":"..."}. Evidence IDs must reference supplied frames. No rankings.
"""
SCHEMA = {'type':'object','additionalProperties':False,'properties':{
    'events':{'type':'array','items':{'type':'object','additionalProperties':False,
        'properties':{'label':{'type':'string','enum':list(CLASSES)},'time_sec':{'type':'number'},
            'evidence_ids':{'type':'array','items':{'type':'string'}},'confidence':{'type':'number'}},
        'required':['label','time_sec','evidence_ids','confidence']}},
    'uncertainty':{'type':'string'}},'required':['events','uncertainty']}


def validate_predictions(data, evidence, duration):
    if not isinstance(data,dict) or set(data)!={'events','uncertainty'} or not isinstance(data['events'],list):
        raise ValueError('Invalid prediction contract')
    if not isinstance(data['uncertainty'],str) or len(data['events'])>200:
        raise ValueError('Unbounded prediction')
    refs={e['id'] for e in evidence if e['kind']=='frame'}
    for row in data['events']:
        if not isinstance(row,dict) or set(row)!={'label','time_sec','evidence_ids','confidence'}:
            raise ValueError('Invalid action prediction')
        if row['label'] not in CLASSES:raise ValueError('Unknown action class')
        for field,upper in (('time_sec',duration),('confidence',1.)):
            value=row[field]
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=upper:
                raise ValueError('Invalid prediction number')
        if row['time_sec']>=duration:raise ValueError('Prediction outside clip')
        if not isinstance(row['evidence_ids'],list) or not row['evidence_ids'] or any(
                not isinstance(ref,str) or ref not in refs for ref in row['evidence_ids']):
            raise ValueError('Missing observed frame reference')
    return data


def match_counts(truth, predictions, tolerance):
    """Maximum cardinality classwise one-to-one matching, then minimum distance."""
    result={}
    for label in CLASSES:
        actual=[r['time_sec'] for r in truth if r['label']==label]
        guessed=[r['time_sec'] for r in predictions if r['label']==label]
        tp=0
        if actual and guessed:
            distances=np.abs(np.asarray(actual)[:,None]-np.asarray(guessed)[None,:])
            valid=distances<=tolerance+1e-9
            penalty=(max(len(actual),len(guessed))+1)*(tolerance+1)
            rows,cols=linear_sum_assignment(np.where(valid,distances,penalty))
            tp=int(valid[rows,cols].sum())
        result[label]={'tp':tp,'fp':len(guessed)-tp,'fn':len(actual)-tp}
    return result


def metrics(counts):
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    return {**counts,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def validate_sample_timing(sample, source):
    """Bind local manifest time units to original labels and probed media."""
    original=sample['annotation_row']
    fps=sample['fps']
    if (type(fps) not in (int,float) or not math.isfinite(fps) or fps<=0
            or fps!=original['fps']):
        raise ValueError('Sample FPS differs from original annotation clock')
    if (type(sample['frames']) is not int or sample['frames']<=0
            or sample['frames']!=source['frame_count']
            or sample['source_declared_num_frames']!=original['num_frames']
            or sample['frames']<original['num_frames']):
        raise ValueError('Sample frame counts differ from original labels or actual media')
    if sample['frame_count_matches_annotation']!=(sample['frames']==original['num_frames']):
        raise ValueError('Source frame discrepancy was not correctly recorded')
    if sample['first_frame_file']!='000000.jpg':
        raise ValueError('Unsupported source frame indexing')
    if sample['match_id']!=original['video'].split('/')[0]:
        raise ValueError('Sample match ID differs from original held-out match')
    for rate in ('nominal_fps','average_fps'):
        if not math.isclose(float(Fraction(source[rate])),fps,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('Encoded video FPS differs from original annotation clock')
    if (not math.isclose(source['duration_sec'],sample['frames']/fps,rel_tol=0,abs_tol=.001)
            or abs(source['start_sec'])>1e-6 or abs(source['video_start_sec'])>1e-6):
        raise ValueError('Encoded video duration or time origin differs from source frames')
    if source['has_audio'] or sample['audio_available'] is not False:
        raise ValueError('This diagnostic requires the documented silent derivative')


def completion_summary(jobs, results, failures):
    expected={job['id'] for job in jobs}
    completed={row['id'] for row in results}
    if len(expected)!=len(jobs) or len(completed)!=len(results) or completed-expected:
        raise ValueError('Duplicate or unexpected request identity')
    per_fps={}
    for fps in sorted({job['fps'] for job in jobs}):
        group={job['id'] for job in jobs if job['fps']==fps}
        relevant=[row for row in failures if row['job'] in group]
        per_fps[str(fps)]={'expected_requests':len(group),'completed_requests':len(group & completed),
                           'complete':group<=completed and not relevant,
                           'missing_request_ids':sorted(group-completed)}
    return {'complete':bool(expected) and completed==expected and not failures,'completion_by_fps':per_fps}


def load_samples(dataset, split='test'):
    if split not in ('train','val','test'):raise ValueError('Use an original author split')
    manifest=read_json(dataset/'manifest.json')
    if manifest['dataset']!='VNL-STES':raise ValueError('Wrong dataset')
    original=read_json(dataset/f'raw/vnl_1.5/{split}.json')
    indexed={r['video']:r for r in original}
    if len(indexed)!=len(original):raise ValueError('Duplicate original test clip IDs')
    match_groups={split:{r['video'].split('/')[0] for r in read_json(dataset/f'raw/vnl_1.5/{split}.json')}
                  for split in ('train','val','test')}
    if any(match_groups[a]&match_groups[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Original match IDs leak across source splits')
    samples=manifest['samples']
    if not samples or len({r['video'] for r in samples})!=len(samples):raise ValueError('Invalid sample coverage')
    for artifact in manifest['original_annotation_files']:
        if digest(dataset/artifact['path'])!=artifact['sha256']:raise ValueError('Source annotation hash changed')
    for sample in samples:
        if sample['split']!=split or sample['annotation_row']!=indexed[sample['video']]:
            raise ValueError('Original held-out test annotation changed')
        if any(e['label'] not in CLASSES or not 0<=e['frame']<sample['frames'] for e in sample['annotation_row']['events']):
            raise ValueError('Original event outside source frame sequence')
        if digest(dataset/sample['derived_video'])!=sample['derived_video_sha256']:
            raise ValueError('Derived video checksum changed')
        validate_sample_timing(sample,probe(dataset/sample['derived_video']))
    return manifest,samples


def evaluate(args):
    start=time.monotonic();deadline=start+args.total_timeout
    manifest,samples=load_samples(args.dataset)
    config=read_json(args.llm_config).get('llm',{})
    endpoint=args.api_base or config.get('base_url')
    model=args.model or config.get('vision_model') or config.get('model')
    key=config.get('api_key') or os.environ.get('VOLLEYMOLE_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not endpoint or not model or not key:raise ValueError('Configure an authorized visual API first')
    if len(set(args.fps))!=len(args.fps):raise ValueError('Sampling rates must be unique')
    jobs=[{'id':f"{row['video'].replace('/','__')}_{fps:.17g}fps",'sample':i,'fps':fps}
          for i,row in enumerate(samples) for fps in args.fps]
    args.output.mkdir(parents=True,exist_ok=True)
    def predict(job):
        began=time.monotonic();row=samples[job['sample']]
        source=probe(args.dataset/row['derived_video'])
        identity={'model':model,'endpoint':endpoint,'fps':job['fps'],'width':args.width,
                  'media_sha256':row['derived_video_sha256'],'prompt':PROMPT,
                  'code_sha256':digest(Path(__file__)),'sampler_sha256':digest(Path(__file__).resolve().parents[1]/'src/volleymole/semantic.py')}
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cache=args.output/'predictions'/f"{job['id']}.json"
        if cache.is_file():
            existing=read_json(cache)
            if existing['signature']==signature:
                validate_predictions(existing['predictions'],existing['evidence'],source['duration_sec'])
                return {**existing,'cached':True}
        evidence,content,_=sampled_evidence(source,0,source['duration_sec'],job['fps'],args.width,deadline=deadline)
        # No annotation rows, event counts, or original labels enter this payload.
        content.insert(0,{'type':'text','text':json.dumps({'duration_sec':source['duration_sec'],
            'audio_available':False,'requested_fps':job['fps'],'actual_frame_times':[e['start_sec'] for e in evidence]})})
        payload={'model':model,'max_tokens':4096,'temperature':0,
                 'messages':[{'role':'system','content':PROMPT},{'role':'user','content':content}],
                 'response_format':{'type':'json_schema','json_schema':{'name':'volleyball_actions','strict':True,'schema':SCHEMA}}}
        remaining=min(args.request_timeout,deadline-time.monotonic())
        if remaining<=0:raise TimeoutError('Benchmark deadline')
        data,metadata=request_json(endpoint,key,payload,remaining)
        result={'id':job['id'],'video':row['video'],'fps':job['fps'],'signature':signature,'identity':identity,
                'predictions':validate_predictions(data,evidence,source['duration_sec']),
                'evidence':evidence,'request':metadata,'elapsed_sec':time.monotonic()-began,'cached':False}
        save_json(cache,result)
        print(f"Completed {job['id']}: {len(evidence)} frames, {result['elapsed_sec']:.1f}s",flush=True)
        return result
    results,failures=bounded_map(jobs,predict,args.concurrency,deadline)
    indexed={r['id']:r for r in results}
    for job in jobs:
        if job['id'] not in indexed and not any(f['job']==job['id'] for f in failures):
            failures.append({'job':job['id'],'error':'NotStartedBeforeDeadline','status':'unreviewed'})
    scores={}
    for fps in args.fps:
        scores[str(fps)]={}
        for tolerance in args.tolerances:
            sums={label:{'tp':0,'fp':0,'fn':0} for label in CLASSES}
            for job in (j for j in jobs if j['fps']==fps):
                row=samples[job['sample']]
                truth=[{'label':r['label'],'time_sec':r['frame']/row['fps']} for r in row['annotation_row']['events']]
                prediction=indexed.get(job['id'],{}).get('predictions',{}).get('events',[])
                for label,counts in match_counts(truth,prediction,tolerance).items():
                    for key_,value in counts.items():sums[label][key_]+=value
            scores[str(fps)][str(tolerance)]={'micro':metrics({k:sum(c[k] for c in sums.values()) for k in ('tp','fp','fn')}),
                                            'per_class':{k:metrics(v) for k,v in sums.items()}}
    report={'benchmark':'VNL-STES action spotting diagnostic','model':model,'fps':args.fps,'width':args.width,
        'test_clips_in_source':manifest['splits']['test']['clips'],'evaluated_sample_clips':len(samples),
        'test_matches':sorted({r['match_id'] for r in samples}),'original_test_sha256':digest(args.dataset/'raw/vnl_1.5/test.json'),
        'selection':'deterministic evenly spaced original test IDs; no action or quality labels used for selection',
        'source_frame_count_discrepancies':[{'video':r['video'],'declared':r['source_declared_num_frames'],'actual':r['frames']}
            for r in samples if not r['frame_count_matches_annotation']],
        'expected_requests':len(jobs),'completed_requests':len(results),'failures':failures,'metrics':scores,
        **completion_summary(jobs,results,failures),
        'elapsed_this_invocation_sec':time.monotonic()-start,'requests':[{k:r[k] for k in ('id','fps','request','elapsed_sec','cached')} for r in results],
        'new_annotations_created':False,'test_set_training_or_prompt_tuning':False,
        'limitations':['Silent reconstructed JPEG sequences; no original audio or full-match offsets.',
            'Author action labels are preserved. Model outputs are predictions only, never new ground truth.',
            'Only a small subset from one held-out match; no confidence claim for all matches.',
            'Fixed-ontology diagnostic of VLM and sampling, not full VolleyMole candidate coverage or dual ranking.',
            'No subjective top-five, humor, explanation correctness, spatial spotting or complete-rally-boundary metric.',
            'Failed and unstarted requests count as empty predictions (false negatives), not silently removed.']}
    save_json(args.output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('data/public_benchmarks/volleyball/vnl_stes'))
    parser.add_argument('--output',type=Path,default=Path('runs/public_benchmarks/vnl_actions'))
    parser.add_argument('--llm-config',type=Path,default=Path('llm_api.json'))
    parser.add_argument('--api-base');parser.add_argument('--model')
    parser.add_argument('--fps',nargs='+',type=float,default=[2.,8.])
    parser.add_argument('--tolerances',nargs='+',type=float,default=[.2,.5,1.])
    parser.add_argument('--width',type=int,default=768)
    parser.add_argument('--concurrency',type=int,choices=range(1,9),default=4)
    parser.add_argument('--request-timeout',type=float,default=120)
    parser.add_argument('--total-timeout',type=float,default=360)
    args=parser.parse_args()
    if any(not math.isfinite(v) or v<=0 for v in args.fps+args.tolerances+[args.request_timeout,args.total_timeout]) or args.width<64:
        parser.error('Sampling, timing and resolution must be finite and positive')
    report=evaluate(args)
    print(json.dumps({'complete':report['complete'],'completed_requests':report['completed_requests'],'failures':report['failures'],'metrics':report['metrics']},ensure_ascii=False))
    if not report['complete']:
        raise SystemExit('Incomplete benchmark; failures remain in report and count as false negatives')


if __name__=='__main__':main()
