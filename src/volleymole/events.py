"""Source-local event facts and deterministic, independently scored collections."""
import copy
import math

WEIGHTS = {
    'highlights': {'action_value': .35, 'attack_defense': .25, 'difficulty_change': .20,
                   'motion_intensity': .10, 'related_reaction': .10},
    'bloopers': {'unexpected_contrast': .35, 'related_laughter': .25,
                 'narrative': .25, 'player_reaction': .15},
}
DIMENSIONS = tuple(k for group in WEIGHTS.values() for k in group)


def number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def windows(duration, length=24., overlap=4.):
    if not number(duration, .001, 1e8) or not 0 <= overlap < length:
        raise ValueError('无效分块参数')
    start = 0.
    while start < duration:
        end = min(duration, start+length)
        yield (start, end)
        if end == duration: break
        start = end-overlap


def validate_events(data, evidence, start, end):
    if not isinstance(data, dict) or set(data) != {'events'} or not isinstance(data['events'], list):
        raise ValueError('事件响应必须包含 events 数组')
    if len(data['events']) > 30:
        raise ValueError('单块事件过多')
    known = {e['id']: e for e in evidence}
    def references(ids, required=False):
        if not isinstance(ids, list) or (required and not ids) or any(not isinstance(i, str) or i not in known for i in ids):
            raise ValueError('事件引用了不存在的证据')
    for event in data['events']:
        required = {'event_type', 'start_sec', 'end_sec', 'clip_start_sec', 'clip_end_sec', 'peak_sec',
            'observations', 'aftermath', 'reactions', 'dimensions', 'uncertainty', 'title',
            'confidence', 'is_rally', 'boundary_complete', 'injury_suspected', 'laughter_linked'}
        if not isinstance(event, dict) or set(event) != required:
            raise ValueError('事件字段不完整')
        for key in ('start_sec', 'end_sec', 'clip_start_sec', 'clip_end_sec', 'peak_sec'):
            if not number(event[key], start, end): raise ValueError('事件时间越过已观察上下文')
        if not start <= event['clip_start_sec'] <= event['start_sec'] < event['end_sec'] <= event['clip_end_sec'] <= end:
            raise ValueError('事件剪辑边界裁断过程')
        if not event['start_sec'] <= event['peak_sec'] <= event['end_sec']:
            raise ValueError('回放峰值不属于事件')
        if not number(event['confidence'], 0, 1): raise ValueError('无效事件置信度')
        for key in ('is_rally', 'boundary_complete', 'injury_suspected', 'laughter_linked'):
            if event[key] is not None and type(event[key]) is not bool: raise ValueError('状态必须为布尔或未知')
        for key, limit in (('event_type', 80), ('title', 40), ('uncertainty', 800)):
            if not isinstance(event[key], str) or not event[key].strip() or len(event[key]) > limit or any(ord(c)<32 for c in event[key]):
                raise ValueError('无效事件说明')
        facts = []
        for key in ('observations', 'aftermath', 'reactions'):
            if not isinstance(event[key], list): raise ValueError('事实必须为数组')
            for fact in event[key]:
                if not isinstance(fact, dict) or set(fact) != {'text', 'time_sec', 'evidence_ids'}:
                    raise ValueError('事实必须带时间和原始证据')
                if not isinstance(fact['text'], str) or not 0 < len(fact['text']) <= 500:
                    raise ValueError('无效事实描述')
                if not number(fact['time_sec'], event['clip_start_sec'], event['clip_end_sec']):
                    raise ValueError('事实不在事件上下文内')
                references(fact['evidence_ids'], True)
                for ref in fact['evidence_ids']:
                    raw = known[ref]
                    if not raw['start_sec']-1 <= fact['time_sec'] <= raw['end_sec']+1:
                        raise ValueError('事实时间与证据不对应')
                facts.append(fact)
        if not event['observations']: raise ValueError('事件没有直接观察')
        cited = {i for f in facts for i in f['evidence_ids']}
        if not any(known[i]['kind'] == 'frame' for i in cited): raise ValueError('事件缺少画面证据')
        if set(event['dimensions']) != set(DIMENSIONS): raise ValueError('双榜维度不完整')
        for key, score in event['dimensions'].items():
            if not isinstance(score, dict) or set(score) != {'value', 'evidence_ids'}:
                raise ValueError('维度必须带证据')
            references(score['evidence_ids'], score['value'] is not None)
            if score['value'] is not None and not number(score['value'], 0, 4): raise ValueError('量表范围为 0–4 或未知')
            if any(i not in cited for i in score['evidence_ids']): raise ValueError('评分必须基于已提取事实')
            if key == 'related_laughter' and score['value'] is not None:
                kinds = {known[i]['kind'] for i in score['evidence_ids']}
                if not kinds.intersection({'audio', 'sound_event'}):
                    raise ValueError('笑声评分缺少音画关联')
                if score['value'] > 0 and (event['laughter_linked'] is not True or 'frame' not in kinds):
                    raise ValueError('笑声评分缺少音画关联')
            if key == 'motion_intensity' and score['value'] is not None:
                if not any(known[i]['kind'] == 'local_motion' and known[i].get('measured') is True for i in score['evidence_ids']):
                    raise ValueError('运动强度缺少可靠相机补偿测量')
    return data['events']


def merge_events(events):
    """Merge duplicate observations, never add scores or treat reviews as evidence."""
    merged = []
    for event in sorted(events, key=lambda e: (e['source_id'], e['start_sec'], e['end_sec'])):
        duplicate = next((e for e in reversed(merged) if e['source_id'] == event['source_id']
            and e['event_type'] == event['event_type'] and abs(e['peak_sec']-event['peak_sec']) <= 2
            and max(0, min(e['end_sec'], event['end_sec'])-max(e['start_sec'], event['start_sec']))
                / min(e['end_sec']-e['start_sec'], event['end_sec']-event['start_sec']) >= .6), None)
        if duplicate is None:
            merged.append(copy.deepcopy(event)); continue
        origins = sorted(set(duplicate.get('origins', [])+event.get('origins', [])))
        if (event.get('review_status') == 'reviewed', event['boundary_complete'] is True, event['confidence']) > (duplicate.get('review_status') == 'reviewed', duplicate['boundary_complete'] is True, duplicate['confidence']):
            merged[merged.index(duplicate)] = duplicate = copy.deepcopy(event)
        duplicate['origins'] = origins
    for i, e in enumerate(merged, 1): e['event_id'] = f'event_{i:05d}'
    return merged


def score_event(event, collection, weights=None):
    weights = WEIGHTS[collection] if weights is None else weights
    if not isinstance(weights, dict) or set(weights) != set(WEIGHTS[collection]) or any(not number(w, 0, 1) for w in weights.values()) or abs(sum(weights.values())-1) > 1e-6:
        raise ValueError('评分权重必须覆盖该榜维度且总和为 1')
    observed = {k: event['dimensions'][k]['value'] for k in weights if event['dimensions'][k]['value'] is not None}
    coverage = sum(weights[k] for k in observed)
    return {'score': round(25*sum(weights[k]*v for k, v in observed.items())/coverage, 3) if coverage else None,
        'observed_weight': round(coverage, 6), 'unknown_dimensions': [k for k in weights if k not in observed],
        'weights': weights, 'dimensions': {k: event['dimensions'][k] for k in weights}}


def select_events(events, collection, count, weights=None, threshold=55.):
    if type(count) is not int or count < 1 or not number(threshold, 0, 100): raise ValueError('无效入选数量或评分阈值')
    pool = []
    for event in events:
        result = score_event(event, collection, weights)
        if result['score'] is None or result['observed_weight'] < .6 or result['score'] < threshold: continue
        if event['boundary_complete'] is not True: continue
        if collection == 'highlights' and event['is_rally'] is not True: continue
        if collection == 'bloopers':
            if event['injury_suspected'] is not False: continue
            if any((event['dimensions'][k]['value'] or 0) < 2 for k in ('unexpected_contrast', 'narrative')): continue
        pool.append((event, result))
    pool.sort(key=lambda pair: (-pair[1]['score'], -pair[1]['observed_weight'], pair[0]['source_id'], pair[0]['start_sec']))
    selected = []
    for event, result in pool:
        if any(event['source_id'] == s['source_id'] and max(event['clip_start_sec'], s['clip_start_sec']) <
               min(event['clip_end_sec'], s['clip_end_sec']) for s, _ in selected): continue
        selected.append((event, result))
        if len(selected) == count: break
    return selected
