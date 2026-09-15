#!/usr/bin/env python3
"""Develop on the original VNL validation match, then evaluate a frozen method.

No original events enter model prompts. This is an action diagnostic, not
human humor scoring. Test runs require a pre-existing validation lockfile.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from volleymole.common import digest,probe,read_json,save_json
from volleymole.semantic import bounded_map,request_json,sampled_evidence
from volleymole.temporal_understanding import action_windows,observe_actions,CLASSES
from evaluate_volleyball_actions import load_samples,match_counts,metrics,validate_predictions,PROMPT,SCHEMA


def experiment_config(args):
    return {'method':args.method,'model':args.model,'fps':2. if args.method=='baseline' else args.fps,
        'endpoint':read_json(args.llm_config)['llm']['base_url'],
        'width':224 if args.method=='baseline' else args.width,'max_frames':args.max_frames,
        'overlap_sec':args.overlap_sec,'threshold':args.threshold,
        'request_timeout':args.request_timeout,'total_timeout':args.total_timeout,'concurrency':args.concurrency,
        'understanding_sha256':digest(ROOT/'src/volleymole/temporal_understanding.py'),
        'baseline_evaluator_sha256':digest(ROOT/'scripts/evaluate_volleyball_actions.py'),
        'comparison_sha256':digest(Path(__file__)),
        'sampler_sha256':digest(ROOT/'src/volleymole/semantic.py')}


def run(args):
    started=time.monotonic();deadline=started+args.total_timeout
    manifest,samples=load_samples(args.dataset,args.split)
    config=experiment_config(args)
    lock=None
    if args.split=='test':
        if args.freeze_from is None:raise ValueError('Test requires --freeze-from a completed validation report')
        lock=read_json(args.freeze_from)
        if (lock['source_split']!='val' or not lock['complete'] or lock['configuration']!=config):
            raise ValueError('Method must be frozen exactly from a complete original validation run')
        if args.output.exists():raise ValueError('Do not overwrite a final test report; preserve the first result')
    llm=read_json(args.llm_config)['llm']
    endpoint=llm['base_url'];key=llm.get('api_key') or os.environ.get('VOLLEYMOLE_API_KEY')
    if not key:raise ValueError('Missing configured API key')
    jobs=[]
    for i,row in enumerate(samples):
        duration=row['frames']/row['fps']
        parts=([{'index':0,'start':0.,'end':duration,'own_start':0.,'own_end':duration}]
            if args.method=='baseline' else action_windows(duration,args.fps,args.max_frames,args.overlap_sec))
        jobs += [{**part,'id':f"sample_{i:04d}_{part['index']:04d}",'sample':i} for part in parts]
    settings={'endpoint':endpoint,'key':key,'model':args.model,'fps':args.fps,'width':args.width,
              'timeout':args.request_timeout,'transport':'video_frames' if args.method=='video_grounded' else 'frames'}
    cache=args.cache/args.method
    def predict(job):
        row=samples[job['sample']];source=probe(args.dataset/row['derived_video'])
        if args.method!='baseline':return {**observe_actions(source,job,cache,settings,deadline),'job':job}
        identity={'source':row['derived_video_sha256'],'config':config,'endpoint':endpoint}
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=cache/f'{signature}.json'
        if path.is_file():
            saved=read_json(path)
            validate_predictions(saved['raw'],saved['evidence'],source['duration_sec'])
            return {**saved,'job':job,'cached':True}
        before=time.monotonic()
        evidence,content,_=sampled_evidence(source,0,source['duration_sec'],2.,224,deadline=deadline)
        content.insert(0,{'type':'text','text':json.dumps({'duration_sec':source['duration_sec'],'audio_available':False,
            'requested_fps':2.,'actual_frame_times':[e['start_sec'] for e in evidence]})})
        payload={'model':args.model,'max_tokens':4096,'temperature':0,
            'messages':[{'role':'system','content':PROMPT},{'role':'user','content':content}],
            'response_format':{'type':'json_schema','json_schema':{'name':'volleyball_actions','strict':True,'schema':SCHEMA}}}
        remaining=min(args.request_timeout,deadline-time.monotonic())
        if remaining<=0:raise TimeoutError('Comparison deadline')
        data,meta=request_json(endpoint,key,payload,remaining)
        result={'signature':signature,'job':job,'events':validate_predictions(data,evidence,source['duration_sec'])['events'],
            'raw':data,'request':meta,'evidence':evidence,'elapsed_sec':time.monotonic()-before,'cached':False}
        save_json(path,result)
        return result
    results,failures=bounded_map(jobs,predict,args.concurrency,deadline)
    completed={r['job']['id'] for r in results};failed={f['job'] for f in failures}
    failures += [{'job':j['id'],'error':'NotStartedBeforeDeadline'} for j in jobs if j['id'] not in completed|failed]
    per_video=[]
    for i,row in enumerate(samples):
        predicted=sorted([event for result in results if result['job']['sample']==i for event in result['events']
                          if event['confidence']>=args.threshold],key=lambda e:e['time_sec'])
        truth=[{'label':e['label'],'time_sec':e['frame']/row['fps']} for e in row['annotation_row']['events']]
        per_video.append({'video':row['video'],'prediction':predicted,
            'original_annotation_sha256':digest(args.dataset/row['original_annotation']),
            'complete':all(j['id'] in completed for j in jobs if j['sample']==i),
            'metrics':{str(t):match_counts(truth,predicted,t) for t in (.2,.5,1.)}})
    totals={}
    for tolerance in ('.2','.5','1.'):
        key_=str(float(tolerance))
        sums={label:{k:sum(row['metrics'][key_][label][k] for row in per_video) for k in ('tp','fp','fn')} for label in CLASSES}
        totals[key_]={'micro':metrics({k:sum(c[k] for c in sums.values()) for k in ('tp','fp','fn')}),
                     'per_class':{label:metrics(c) for label,c in sums.items()}}
    report={'source_split':args.split,'configuration':config,'complete':not failures and len(results)==len(jobs),
        'source_manifest_sha256':digest(args.dataset/'manifest.json'),'clips':len(samples),'matches':sorted({r['match_id'] for r in samples}),
        'expected_requests':len(jobs),'completed_requests':len(results),'failures':failures,'metrics':totals,'per_video':per_video,
        'elapsed_sec':time.monotonic()-started,'original_annotations_changed':False,'new_annotations':False,
        'frozen_from_sha256':digest(args.freeze_from) if lock else None,
        'sampling':{'client_frame_count':sum(len(r['evidence']) for r in results),
                    'server_native_video_sampling_verified':False if args.method=='video_grounded' else None},
        'limitations':['Single match in this split; not a multi-match confidence bound.',
                       'Model predictions are not ground truth or confirmed action outcomes.',
                       'Failed contexts retain all original false negatives; no selective omission.',
                       'Diagnostic action ontology differs from full semantic ranking and humor quality.']}
    save_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('source_split','configuration','complete','expected_requests','completed_requests','failures','metrics')},ensure_ascii=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--method',choices=('baseline','grounded','video_grounded'),default='grounded')
    p.add_argument('--model',default='qwen3-vl-8b-instruct')
    p.add_argument('--llm-config',type=Path,default=ROOT/'llm_api.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=ROOT/'runs/action_accuracy/vlm_cache')
    p.add_argument('--freeze-from',type=Path)
    p.add_argument('--fps',type=float,default=8.)
    p.add_argument('--width',type=int,default=398)
    p.add_argument('--max-frames',type=int,default=24)
    p.add_argument('--overlap-sec',type=float,default=.5)
    p.add_argument('--threshold',type=float,default=0.)
    p.add_argument('--concurrency',type=int,choices=range(1,9),default=4)
    p.add_argument('--request-timeout',type=float,default=90)
    p.add_argument('--total-timeout',type=float,default=600)
    args=p.parse_args()
    import math
    if (not math.isfinite(args.threshold) or not 0<=args.threshold<=1 or args.width<64
            or not math.isfinite(args.request_timeout) or args.request_timeout<=0
            or not math.isfinite(args.total_timeout) or args.total_timeout<=0):p.error('Invalid numerical configuration')
    action_windows(1.,args.fps,args.max_frames,args.overlap_sec)
    report=run(args)
    if not report['complete']:raise SystemExit(2)


if __name__=='__main__':main()
