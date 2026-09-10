"""Visual evidence for text-only ranking models, with resumable small batches."""
import base64
import hashlib
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen
from .common import APP, digest, read_json, save_json
from .schemas import local_file

REVIEW_FIELDS = {
    'rally_id': {'type': 'string'},
    'observation': {'type': 'string'},
    'uncertainty': {'type': 'string'},
    'watchability_score': {'type': 'number'},
    'confidence': {'type': 'number'},
}
REVIEW_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'properties': {'reviews': {'type': 'array', 'items': {'type': 'object',
        'additionalProperties': False, 'properties': REVIEW_FIELDS, 'required': list(REVIEW_FIELDS)}}},
    'required': ['reviews']}


class ResponseContractError(ValueError):
    """Safe diagnostics without retaining remote response bodies or credentials."""
    def __init__(self, reason, finish_reason=None):
        super().__init__(reason)
        self.reason = reason
        self.finish_reason = finish_reason if finish_reason in ('stop','length','content_filter','tool_calls',None) else 'other'


def request_json(endpoint, key, payload, timeout):
    request = Request(endpoint.rstrip('/')+'/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    choice = data['choices'][0]
    if choice['message'].get('refusal'):
        raise ResponseContractError('refused',choice.get('finish_reason'))
    if choice.get('finish_reason') != 'stop':
        raise ResponseContractError('incomplete_response',choice.get('finish_reason'))
    try:
        parsed = json.loads(choice['message']['content'])
    except (json.JSONDecodeError,TypeError):
        raise ResponseContractError('invalid_json','stop') from None
    return parsed, {
        'model': payload['model'], 'finish_reason': choice['finish_reason'], 'usage': data.get('usage')}


def candidate_fields(rally, full_evidence=False):
    fields = {**{k:rally[k] for k in ('rally_id', 'start_sec', 'end_sec', 'safe_start_sec', 'safe_end_sec',
                                'duration_sec', 'actions', 'players', 'ball_metrics', 'rule_score')},
            'uncertainty':rally.get('uncertainty',{'note':'No extra fusion audit in this historical candidate.'})}
    if full_evidence or 'uncertainty' not in rally:
        return fields
    raw = rally['uncertainty']
    gaps = raw.get('occlusion_gaps',[])
    associated = raw.get('auxiliary_associations',[])
    # Frame-by-frame associations stay in the manifest. Bound API context while
    # retaining counts, the longest missing intervals and explicit uncertainty.
    fields['uncertainty'] = {
        'low_ball_dominant':raw.get('low_ball_dominant'),
        'associated_detection_count':len(associated),
        'association_examples':[{k:a[k] for k in ('frame','time_sec','source','vball_anchors') if k in a}
                                for a in associated[:3]],
        'occlusion_gap_count':len(gaps),
        'longest_occlusion_gaps':sorted(gaps,key=lambda g:g['end_sec']-g['start_sec'],reverse=True)[:3],
        'boundary_resolution_sec':raw.get('boundary_resolution_sec'),
        'conservative_boundary_guards_sec':raw.get('conservative_boundary_guards_sec',[])[:4],
        'note':raw.get('note','Uncalibrated model evidence; not confirmed score or winner.'),
        'detail_policy':'Bounded summary; complete raw evidence remains in the local match manifest.'}
    return fields


def image_parts(rally, directory):
    parts = []
    for i, name in enumerate(rally['preview_frames'][:3]):
        when = rally.get('preview_times_sec', [None]*3)[i]
        parts.append({'type': 'text', 'text': f"{rally['rally_id']} / {('开局','动作峰值','结束')[i]} / 源时间 {when} 秒"})
        path = local_file(directory, name)
        parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+
                      base64.b64encode(path.read_bytes()).decode(), 'detail': 'low'}})
    return parts


def choose_vision_model(endpoint, key, timeout):
    request = Request(endpoint.rstrip('/')+'/models', headers={'Authorization': 'Bearer '+key})
    with urlopen(request, timeout=min(timeout, 40)) as response:
        models = [m['id'] for m in json.load(response)['data']]
    # Only use models advertised by this same authorized provider. Image
    # generators, speech and embedding models are never selected as reviewers.
    def priority(name):
        n = name.lower()
        if 'qwen' in n and ('vl' in n or n.startswith(('qwen3.5','qwen3.6','qwen3.8'))): return 0
        if n.startswith('kimi-'): return 1
        if 'glm' in n and ('-v' in n or n.endswith('v')): return 2
        return 99
    candidates = sorted((m for m in models if priority(m)<99), key=lambda m:(priority(m), m))
    if not candidates:
        raise ValueError('该服务没有可自动选择的视觉模型；可用 --vision-model 显式指定')
    return candidates[0], models


def validate_reviews(data, batch):
    if not isinstance(data, dict) or set(data) != {'reviews'} or not isinstance(data['reviews'], list):
        raise ValueError('视觉评审必须符合固定 schema')
    expected = {r['rally_id'] for r in batch}
    seen = set()
    for r in data['reviews']:
        if not isinstance(r, dict) or set(r) != set(REVIEW_FIELDS):
            raise ValueError('视觉评审字段不完整')
        if not isinstance(r['rally_id'], str) or r['rally_id'] not in expected or r['rally_id'] in seen:
            raise ValueError('视觉评审包含未知或重复回合')
        seen.add(r['rally_id'])
        for key, upper in [('watchability_score', 100), ('confidence', 1)]:
            if type(r[key]) not in (int,float) or not math.isfinite(r[key]) or not 0<=r[key]<=upper:
                raise ValueError('视觉评分必须是范围内的有限数值')
        for key in ('observation', 'uncertainty'):
            if not isinstance(r[key], str) or not r[key].strip() or len(r[key])>800:
                raise ValueError('视觉证据说明为空或过长')
    if seen != expected:
        raise ValueError('视觉评审没有覆盖本批全部候选')
    return data['reviews']


def visual_reviews(candidates, directory, endpoint, model, key, timeout, batch_size=1):
    directory = Path(directory)
    prompt = (APP/'prompts/review_frames.md').read_text(encoding='utf-8')
    reviews, artifacts, batches = [], [], []
    for start in range(0,len(candidates),batch_size):
        batch = candidates[start:start+batch_size]
        path = directory/'semantic'/f'batch_{start//batch_size+1:02d}.json'
        signature = hashlib.sha256(json.dumps({'endpoint':endpoint, 'model':model, 'prompt':prompt,
            # Fingerprint all original evidence, even when the network payload
            # is summarized. Existing reviews of the same richer evidence remain
            # usable; changed observations/images/models/prompts still invalidate.
            'candidates':[candidate_fields(r,full_evidence=True) for r in batch],
            'previews':[(r['preview_times_sec'], [digest(local_file(directory,p)) for p in r['preview_frames'][:3]]) for r in batch]},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = read_json(path) if path.exists() else None
        if cached and cached.get('signature') == signature:
            current = validate_reviews({'reviews':cached['reviews']},batch)
            print(f'[vision] 复用批次 {start//batch_size+1}', flush=True)
        else:
            print(f'[vision] {model} 评审候选 {start+1}–{start+len(batch)} / {len(candidates)}', flush=True)
            content = []
            for r in batch:
                content.append({'type':'text','text':json.dumps(candidate_fields(r),ensure_ascii=False)})
                content.extend(image_parts(r,directory))
            payload = {'model':model, 'max_tokens':4096, 'messages':[{'role':'system','content':prompt},{'role':'user','content':content}],
                'response_format':{'type':'json_schema','json_schema':{'name':'rally_visual_reviews','strict':True,'schema':REVIEW_SCHEMA}}}
            data, metadata = request_json(endpoint,key,payload,timeout)
            current = validate_reviews(data,batch)
            cached = {'signature':signature,'reviews':current,'request':metadata,'evidence_format':'bounded_summary_v1',
                      'candidate_ids':[r['rally_id'] for r in batch],'image_count':3*len(batch)}
            save_json(path,cached)
        reviews.extend(current);artifacts.append(path);batches.append(cached['request'])
    return reviews, artifacts, batches
