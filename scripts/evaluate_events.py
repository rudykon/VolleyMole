#!/usr/bin/env python3
"""Evaluate held-out, match-grouped annotations and like-for-like wall times.

Input JSON: {matches: [{match_id, split, annotations: [{collection, source_id,
start_sec, end_sec, clip_start_sec, clip_end_sec}], predictions: {events: [...],
selected: {highlights: [...], bloopers: [...]}}, adjudications: {event_id:
{quality: 0..4, explanation_correct: true|false}}}], timings: [{variant,
hardware, output_spec, cache_condition, total_elapsed_sec}]}. Selected rows
use event_id/source_id/start_sec/end_sec/clip_start_sec/clip_end_sec.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load_predictions(directory):
    directory=Path(directory)
    timeline=json.loads((directory/'event_timeline.json').read_text())
    selected={}
    for collection in ('highlights','bloopers'):
        folder=directory/'collections'/collection
        if not (folder/'edit_decision.json').is_file():continue
        decision=json.loads((folder/'edit_decision.json').read_text())
        manifest=json.loads((folder/'match_manifest.json').read_text())
        by_id={r['rally_id']:r for r in manifest['rallies']}
        selected[collection]=[{'event_id':item['rally_id'],'source_id':by_id[item['rally_id']].get('source_id','single'),
            'start_sec':by_id[item['rally_id']]['start_sec'],'end_sec':by_id[item['rally_id']]['end_sec'],
            'clip_start_sec':item['clip_start_sec'],'clip_end_sec':item['clip_end_sec']} for item in decision['selected']]
    return {'events':timeline['events'],'selected':selected}


def overlap(a,b):
    if a.get('source_id','single') != b.get('source_id','single'):return 0.
    intersection=max(0,min(a['end_sec'],b['end_sec'])-max(a['start_sec'],b['start_sec']))
    union=max(a['end_sec'],b['end_sec'])-min(a['start_sec'],b['start_sec'])
    return intersection/union if union>0 else 0.


def evaluate(data, split='test', iou=.5):
    results=[];seen=set()
    for match in data['matches']:
        # One match belongs to exactly one partition: never calibrate on test.
        key=match['match_id']
        if key in seen:raise ValueError('同一比赛不能重复或跨数据分区')
        seen.add(key)
        if match['split']!=split:continue
        predicted=load_predictions(match['run_directory']) if 'run_directory' in match else match['predictions']
        judgments=match.get('adjudications',{})
        for collection in ('highlights','bloopers'):
            truth=[a for a in match['annotations'] if a['collection']==collection]
            events=predicted['events']
            covered=sum(any(overlap(a,e)>=iou for e in events) for a in truth)
            selected=predicted['selected'].get(collection,[])[:5]
            grades=[judgments[e['event_id']]['quality'] for e in selected if e['event_id'] in judgments]
            correctness=[judgments[e['event_id']]['explanation_correct'] for e in selected if e['event_id'] in judgments]
            complete=[]
            for e in selected:
                matching=sorted(truth,key=lambda a:overlap(a,e),reverse=True)
                if matching and overlap(matching[0],e)>=iou:
                    a=matching[0]
                    complete.append(e['clip_start_sec']<=a['clip_start_sec'] and e['clip_end_sec']>=a['clip_end_sec'])
            if any(type(g) not in (int,float) or not 0<=g<=4 for g in grades):raise ValueError('人工质量量表为 0–4')
            if any(type(c) is not bool for c in correctness):raise ValueError('解释正确性须由人工标注为布尔')
            results.append({'match_id':key,'collection':collection,'annotated_events':len(truth),
                'candidate_recall':covered/len(truth) if truth else None,
                'selected_count':len(selected),'judged_count':len(grades),
                'top5_quality':float(np.mean(grades)) if grades else None,
                'explanation_accuracy':float(np.mean(correctness)) if correctness else None,
                'clip_completeness':float(np.mean(complete)) if complete else None,
                'matched_selected_count':len(complete)})
    groups=defaultdict(lambda:defaultdict(list))
    for row in data.get('timings',[]):
        key=tuple(row[k] for k in ('hardware','output_spec','cache_condition'))
        elapsed=row['total_elapsed_sec']
        if not isinstance(elapsed,(int,float)) or not np.isfinite(elapsed) or elapsed<=0:raise ValueError('完整耗时必须为正有限数')
        groups[key][row['variant']].append(elapsed)
    timings=[]
    for key,variants in groups.items():
        summary={v:{'n':len(values),'median_sec':float(np.median(values)),'p95_sec':float(np.percentile(values,95))}
                 for v,values in variants.items()}
        comparison={}
        if 'baseline' in summary and 'events' in summary:
            for percentile in ('median_sec','p95_sec'):
                ratio=summary['events'][percentile]/summary['baseline'][percentile]
                comparison[percentile]={'ratio':ratio,'within_initial_25_percent_target':ratio<=1.25}
        timings.append({'hardware':key[0],'output_spec':key[1],'cache_condition':key[2],
                        'variants':summary,'comparison':comparison})
    return {'split':split,'iou_threshold':iou,'matches':results,'timings':timings,
            'note':'Unknown judgments are omitted and counted, never replaced with zero; 25% is an empirical optimization target.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',default='test');parser.add_argument('--iou',type=float,default=.5)
    args=parser.parse_args()
    if not 0<args.iou<=1:parser.error('--iou 必须在 (0,1]')
    report=evaluate(json.loads(args.input.read_text()),args.split,args.iou)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
