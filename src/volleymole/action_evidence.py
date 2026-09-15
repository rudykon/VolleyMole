"""Import source-bound local action proposals as hypotheses, not new facts."""
import math
from pathlib import Path
import re

from .common import digest,read_json

LABELS={'serve','receive','set','spike','block','score'}


def validate_action_evidence(data,source):
    if (not isinstance(data,dict) or type(data.get('schema_version')) is not int or data['schema_version']!=1
            or data.get('status') not in ('complete','partial','unsupported')):
        raise ValueError('动作候选文件版本或状态无效')
    origin=data.get('source',{})
    if not isinstance(origin,dict) or origin.get('time_basis')!='source_relative_to_container_start':
        raise ValueError('动作候选须使用相对源容器起点的时间')
    if origin.get('sha256')!=source['identity']['sha256']:
        raise ValueError('动作候选不是当前源视频；不能复用其他录像的预测')
    duration=origin.get('duration_sec')
    if (type(duration) not in (int,float) or not math.isfinite(duration)
            or abs(duration-source['duration_sec'])>.05):raise ValueError('动作候选源时间基准不匹配')
    model=data.get('model',{})
    if not isinstance(model,dict) or any(not isinstance(model.get(k),str) or not re.fullmatch('[0-9a-f]{64}',model[k])
           for k in ('sha256','manifest_sha256')):raise ValueError('动作候选缺少模型来源指纹')
    events=data.get('events')
    if not isinstance(events,list) or len(events)>1_000_000:raise ValueError('动作候选列表无效')
    if data['status']=='unsupported' and events:raise ValueError('不支持的输入不能包含动作预测')
    covered_start=data.get('covered_start_sec')
    covered_end=data.get('covered_end_sec')
    if data['status']=='partial' and not events and covered_start is None and covered_end is None:
        return data
    if data['status']!='unsupported' and (type(covered_start) not in (int,float)
            or type(covered_end) not in (int,float) or not math.isfinite(covered_start) or not math.isfinite(covered_end)
            or not 0<=covered_start<=covered_end<=duration+.05):raise ValueError('动作候选缺少有效已覆盖区间')
    for row in events:
        if not isinstance(row,dict) or row.get('label') not in LABELS:raise ValueError('动作类别无效')
        t,p=row.get('time_sec'),row.get('raw_probability')
        if (type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<duration
                or type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1):
            raise ValueError('动作候选时间或原始概率无效')
        if row.get('observation_status')!='candidate':raise ValueError('动作模型输出只能导入为待核实候选')
        if not covered_start<=t<covered_end:raise ValueError('动作候选越过实际模型覆盖区间')
        if row.get('model_sha256',model['sha256'])!=model['sha256']:raise ValueError('动作候选模型指纹不一致')
    return data


def evidence_status(data):
    """Preserve the distinction between no detections and unprocessed footage."""
    return {k:data.get(k) for k in ('status','covered_start_sec','covered_end_sec',
                                   'processed_frames','finalized_frames','reason','model')}


def load_action_evidence(args,source):
    path=getattr(args,'action_evidence',None)
    directory=getattr(args,'action_evidence_dir',None)
    if directory:path=Path(directory)/(source['identity']['sha256']+'.json')
    if path is None:return {'status':'not_configured','events':[]},None
    path=Path(path)
    if not path.is_file():
        if directory:return {'status':'not_found','events':[]},None
        raise ValueError('动作候选文件不存在')
    return validate_action_evidence(read_json(path),source),digest(path)


def context_hypotheses(data,start,end):
    # No evidence IDs: a classifier label is not an independent observation of
    # the pixels. Semantic facts must still cite actual video/audio evidence.
    return [{'label':r['label'],'time_sec':r['time_sec'],'raw_probability':r['raw_probability'],
             'status':'unverified_visual_model_hypothesis'}
            for r in data.get('events',[]) if start<=r['time_sec']<end]


def uncovered_action_contexts(data,covered,duration,padding=2.):
    times=sorted({r['time_sec'] for r in data.get('events',[])
                  if not any(j['start']<=r['time_sec']<j['end'] for j in covered)})
    result=[]
    for t in times:
        a,b=max(0.,t-padding),min(duration,t+padding)
        if result and a<result[-1]['end'] and b-result[-1]['start']<=12.:
            result[-1]['end']=b
        else:result.append({'start':a,'end':b})
    return [{**span,'id':f'local_action_{i:05d}','phase':'review',
             'candidate':{'origin':'unverified_local_action_model','hypotheses':context_hypotheses(data,span['start'],span['end'])}}
            for i,span in enumerate(result)]
