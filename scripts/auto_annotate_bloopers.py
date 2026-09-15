#!/usr/bin/env python3
"""Create explicitly model-generated, visual-only blooper annotations.

These are not human judgments or ground truth, and never train a model here.
Inputs longer than 12 seconds are recorded as unsupported; no tail is dropped.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from volleymole.common import digest,probe,read_json,save_json
from volleymole.semantic import bounded_map,request_json,sampled_evidence
from volleymole.blooper_benchmark import DIMENSIONS,FLAGS,STAGES

VERSION='volleymole-bloopers-model-visual-v1'
PROMPT='''You annotate volleyball video for a possible funny-moments collection.
Use only the supplied ordered visual frames and their explicit source_sec times.
There is NO audio evidence. Do not infer laughter or sounds from facial expressions.
This is automatic model annotation, never human ground truth. Do not force positives.
Ordinary errors, falls, injuries, pain, celebrations and missing views are not
automatically funny. Do not invent contact, success, score, intention or outcome.
Describe ordinary visible play too: visible facts do NOT require humor. If ordinary play is clearly visible without a funny contrast, record that factual scene with actual frame IDs and rate overall_fun=0 and selection_suitability=0; other unsupported fields remain null. Reserve empty facts and all-null ratings for genuinely unobservable/occluded evidence, not merely absence of humor. Write fact descriptions and uncertainty in Chinese.
First describe visible facts with actual supplied frame IDs; then rate what those
facts support. Missing evidence is null, not zero. A short ordinary rally may have
zero fun. Full narrative needs visible setup, unexpected development and reaction.
Ratings are integers 0..4 or null. overall_fun: 0 not funny,1 weak,2 moderate,
3 clearly funny,4 unusually funny. selection_suitability: 0 unsuitable,1 weak,
2 possible,3 good choice,4 exceptional. unexpected_contrast: 0 none/ordinary,
1 weak,2 visible contrast,3 clear surprising mismatch,4 strong unexpected contrast.
narrative: 0 unintelligible,1 important stages absent,2 understandable but partial,
3 complete cause/outcome,4 complete setup/unexpected/reaction. player_reaction:
0 clearly none,1 minor,2 visible individual,3 clear teammates,4 strong group reaction.
related_laughter MUST be null. laughter_linked and unrelated_laughter MUST be null.
injury_suspected is true for visible possible pain/injury, false only if visual
evidence supports no such concern, otherwise null. safe_to_include cannot be true
unless injury_suspected=false and event_visible=true. Do not fill missing stages.
Return only JSON with EXACT keys:
{"observed_facts":[{"text":"visible fact","frame_ids":["frame_00000"]}],
"ratings":{"overall_fun":null,"selection_suitability":null,"unexpected_contrast":null,
"related_laughter":null,"narrative":null,"player_reaction":null},
"flags":{"event_visible":null,"boundary_complete":null,"injury_suspected":null,
"safe_to_include":null,"laughter_linked":null,"ordinary_error":null,"unrelated_laughter":null},
"stages":{"setup":{"start_sec":null,"end_sec":null},
"unexpected":{"start_sec":null,"end_sec":null},"reaction":{"start_sec":null,"end_sec":null}},
"confidence":0.0,"uncertainty":"Explain missing views, sampling and interpretation limits."}
Stage bounds must use source_sec within the supplied interval, in narrative order.
If no facts are reliably visible, return empty facts and all ratings/flags/stages
null, confidence zero, with the reason in uncertainty. Do not return rankings.
'''


def number(v,lo,hi):
    return type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi


def validate_prediction(value,evidence,start,end):
    keys={'observed_facts','ratings','flags','stages','confidence','uncertainty'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('prediction_contract')
    if not isinstance(value['observed_facts'],list) or len(value['observed_facts'])>40:raise ValueError('facts_contract')
    refs={r['id'] for r in evidence if r['kind']=='frame'}
    for row in value['observed_facts']:
        if not isinstance(row,dict) or set(row)!={'text','frame_ids'}:raise ValueError('fact_contract')
        if not isinstance(row['text'],str) or not 0<len(row['text'].strip())<=600:raise ValueError('fact_text')
        if (not isinstance(row['frame_ids'],list) or not row['frame_ids']
                or any(not isinstance(r,str) or r not in refs for r in row['frame_ids'])):raise ValueError('invented_frame_reference')
    for field,expected in (('ratings',DIMENSIONS),('flags',FLAGS),('stages',STAGES)):
        if not isinstance(value[field],dict) or set(value[field])!=set(expected):raise ValueError(field+'_contract')
    if any(v is not None and (type(v) is not int or not 0<=v<=4) for v in value['ratings'].values()):raise ValueError('rating_range')
    if any(v is not None and type(v) is not bool for v in value['flags'].values()):raise ValueError('flag_range')
    if value['ratings']['related_laughter'] is not None or any(value['flags'][k] is not None for k in ('laughter_linked','unrelated_laughter')):
        raise ValueError('visual_only_cannot_label_laughter')
    known=[]
    for stage in STAGES:
        span=value['stages'][stage]
        if not isinstance(span,dict) or set(span)!={'start_sec','end_sec'}:raise ValueError('stage_contract')
        a,b=span['start_sec'],span['end_sec']
        if a is None and b is None:continue
        if not number(a,start,end) or not number(b,start,end) or a>b:raise ValueError('stage_outside_source_clock')
        known.append((a,b))
    if any(a[0]>b[0] for a,b in zip(known,known[1:])):raise ValueError('stage_order')
    if not number(value['confidence'],0,1):raise ValueError('confidence_range')
    if not isinstance(value['uncertainty'],str) or not 0<len(value['uncertainty'].strip())<=2000:raise ValueError('uncertainty_required')
    if value['flags']['safe_to_include'] is True and (value['flags']['injury_suspected'] is not False or value['flags']['event_visible'] is not True):
        raise ValueError('safety_uncertainty')
    if not value['observed_facts'] and (known or any(v is not None for v in value['ratings'].values())
            or any(v is not None for v in value['flags'].values()) or value['confidence']!=0):raise ValueError('unsupported_empty_facts')
    return value


def validate_record(record):
    if (record.get('schema_version')!=VERSION or record.get('annotation_source')!='model'
            or record.get('is_ground_truth') is not False or record.get('human_only') is not False):
        raise ValueError('automatic_annotation_cannot_claim_human_or_ground_truth')
    return validate_prediction(record['annotation'],record['evidence'],record['start_sec'],record['end_sec'])


def run(args):
    began=time.monotonic();deadline=began+args.total_timeout
    data=read_json(args.input);items=data.get('items')
    if not isinstance(items,list) or not items:raise ValueError('Nonempty input items required')
    ids=[r.get('id') for r in items]
    if any(not isinstance(i,str) or not i.strip() for i in ids) or len(set(ids))!=len(ids):raise ValueError('Unique item IDs required')
    config=read_json(args.llm_config).get('llm',{})
    endpoint=args.api_base or config.get('base_url')
    key=config.get('api_key') or os.getenv('VOLLEYMOLE_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not endpoint or not key:raise ValueError('Configure the authorized visual API')
    args.output.mkdir(parents=True,exist_ok=True)
    parameters={'model':args.model,'endpoint':endpoint,'fps':2,'width':224,'max_frames':24,
        'max_interval_sec':12,'max_tokens':2048,'concurrency':args.concurrency,
        'total_timeout_sec':args.total_timeout,'request_timeout_sec':args.request_timeout,
        'audio_used':False,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
        'script_sha256':digest(Path(__file__)),'sampler_sha256':digest(ROOT/'src/volleymole/semantic.py')}
    save_json(args.output/'protocol.json',{'schema_version':VERSION,'annotation_source':'model',
        'is_ground_truth':False,'human_only':False,'parameters':parameters,'prompt':PROMPT,
        'input_sha256':digest(args.input),'purpose':'automatic silver annotations; not human evaluation or training truth'})
    failures=[];jobs=[];sources={}
    for row in items:
        try:
            if not isinstance(row.get('match_id'),str) or not isinstance(row.get('source_split'),str):raise ValueError('source_provenance_required')
            source_path=str(Path(row['video']).resolve())
            if source_path not in sources:
                sources[source_path]=probe(source_path)
                sources[source_path]['identity']={'sha256':digest(source_path)}
            source=sources[source_path]
            a,b=row['start_sec'],row['end_sec']
            if not number(a,0,source['duration_sec']) or not number(b,0,source['duration_sec']) or a>=b:raise ValueError('invalid_source_interval')
            if b-a>12:raise ValueError('unsupported_interval_over_12_seconds_no_truncation')
            jobs.append({'id':row['id'],'item':row,'source':source})
        except Exception as exc:
            failures.append({'job':row['id'],'status':'unannotated','error':type(exc).__name__,
                             'reason':str(exc) if isinstance(exc,ValueError) else 'source_verification_failed'})
    def annotate(job):
        row,source=job['item'],job['source'];a,b=row['start_sec'],row['end_sec']
        evidence,content,_=sampled_evidence(source,a,b,2,224,None,deadline)
        if not 1<=len(evidence)<=24:raise ValueError('sample_frame_budget_or_coverage')
        # Only times and pixels are sent: no prior rank, source action labels,
        # match identity, original split or event-selection metadata.
        content.insert(0,{'type':'text','text':json.dumps({'source_interval_sec':[a,b],
            'audio_available_to_annotator':False,'sample_times_sec':[e['start_sec'] for e in evidence],
            'frame_ids':[e['id'] for e in evidence]})})
        payload={'model':args.model,'temperature':0,'max_tokens':2048,
            'messages':[{'role':'system','content':PROMPT},{'role':'user','content':content}],
            'response_format':{'type':'json_object'}}
        remaining=min(args.request_timeout,deadline-time.monotonic())
        if remaining<=0:raise TimeoutError('annotation_deadline')
        observed,metadata=request_json(endpoint,key,payload,remaining)
        validate_prediction(observed,evidence,a,b)
        result={'schema_version':VERSION,'annotation_source':'model','is_ground_truth':False,'human_only':False,
            'id':row['id'],'match_id':row['match_id'],'source_split':row['source_split'],
            'source':{'path':str(Path(row['video']).resolve()),'sha256':source['identity']['sha256'],
                      'duration_sec':source['duration_sec'],'has_audio':source['has_audio']},
            'start_sec':a,'end_sec':b,'annotation':observed,'evidence':evidence,
            'model':args.model,'parameters':parameters,'request':metadata,'status':'complete'}
        validate_record(result)
        save_json(args.output/'annotations'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json'),result)
        print(f"Annotated {row['id']} ({len(evidence)} frames)",flush=True)
        return result
    results,failed=bounded_map(jobs,annotate,args.concurrency,deadline)
    failures.extend(failed)
    done={r['id'] for r in results};failed_ids={r['job'] for r in failures}
    failures.extend({'job':r['id'],'status':'unannotated','error':'deadline_before_request'} for r in jobs if r['id'] not in done|failed_ids)
    save_json(args.output/'annotations.json',{'schema_version':VERSION,'annotation_source':'model',
        'is_ground_truth':False,'human_only':False,'items':results})
    save_json(args.output/'failures.json',failures)
    summary={'schema_version':VERSION,'annotation_source':'model','is_ground_truth':False,'human_only':False,
        'status':'complete' if len(results)==len(items) and not failures else 'partial',
        'requested_items':len(items),'annotated_items':len(results),'failed_items':len(failures),
        'visual_only':True,'laughter_labels_known':0,'new_human_annotations':0,'used_for_training':False,
        'potential_fun_items':sum((r['annotation']['ratings']['overall_fun'] or 0)>=2 for r in results),
        'elapsed_sec':time.monotonic()-began,'limitations':['Automatic model judgments are fallible silver labels, not evaluation ground truth.',
            'No audio interpretation; every laughter field remains unknown.',
            'Selected short intervals do not establish full-match blooper recall or a human benchmark.']}
    save_json(args.output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=4,choices=range(1,9))
    parser.add_argument('--total-timeout',type=float,default=360);parser.add_argument('--request-timeout',type=float,default=45)
    parser.add_argument('--model',default='qwen3-vl-8b-instruct');parser.add_argument('--llm-config',type=Path,default=Path('llm_api.json'))
    parser.add_argument('--api-base')
    args=parser.parse_args()
    if any(not math.isfinite(v) or v<=0 for v in (args.total_timeout,args.request_timeout)):parser.error('Finite positive timeouts required')
    print(json.dumps(run(args),ensure_ascii=False))


if __name__=='__main__':main()
