"""Wire contract for event understanding; semantic evidence checks stay local."""
from .events import DIMENSIONS, validate_events

WIRE_PROTOCOL = 'frame_anchors_v3'
FRAME_FIELDS = ('start_frame', 'end_frame', 'clip_start_frame', 'clip_end_frame', 'peak_frame')
EVENT_FIELDS = ('event_type', 'title', 'confidence', 'is_rally', 'boundary_complete',
                'injury_suspected', 'laughter_linked', 'uncertainty')
FACT_KINDS = {'observation': 'observations', 'aftermath': 'aftermath', 'reaction': 'reactions'}
MAX_EVENTS = 3
MAX_FACTS = 12


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def wire_event_response_format(evidence, start, end):
    """V3: emit frame anchors and explicitly named dimension array entries.

    The canonical event contract below remains unchanged. Cross-field ordering
    and actual evidence membership are checked again by decode_event_response;
    dimension names, never array positions, determine the canonical scores.
    """
    frames = [e['id'] for e in evidence if e['kind'] == 'frame']
    supports = [e['id'] for e in evidence if e['kind'] in ('audio', 'sound_event', 'local_motion')]
    if not frames:
        raise ValueError('事件理解缺少画面证据')
    text = lambda limit: {'type': 'string', 'minLength': 1, 'maxLength': limit}
    frame = {'$ref': '#/$defs/frame_id'}
    index_list = {'type': 'array', 'maxItems': MAX_FACTS,
                  'items': {'type': 'integer', 'minimum': 0, 'maximum': MAX_FACTS-1}}
    known_dimensions = list(DIMENSIONS)
    if not any(e['kind'] in ('audio', 'sound_event') for e in evidence):
        known_dimensions.remove('related_laughter')
    if not any(e['kind'] == 'local_motion' and e.get('measured') is True for e in evidence):
        known_dimensions.remove('motion_intensity')
    unknown = _object({'dimension': {'type': 'string', 'enum': list(DIMENSIONS)},
        'value': {'type': 'null'}, 'fact_indexes': {**index_list, 'maxItems': 0}})
    score = {'anyOf': [unknown, _object({
        'dimension': {'type': 'string', 'enum': known_dimensions},
        'value': {'type': 'number', 'minimum': 0, 'maximum': 4},
        'fact_indexes': {**index_list, 'minItems': 1}})]}
    fact = _object({'kind': {'type': 'string', 'enum': list(FACT_KINDS)}, 'text': text(300),
        'frame_id': frame, 'support_id': {'anyOf': [{'type': 'null'},
            {'type': 'string', 'enum': supports}]} if supports else {'type': 'null'}})
    item = _object({'event_type': text(80), **{key: frame for key in FRAME_FIELDS},
        'title': text(40), 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        **{key: {'type': ['boolean', 'null']} for key in
           ('is_rally', 'boundary_complete', 'injury_suspected', 'laughter_linked')},
        'uncertainty': text(800),
        'facts': {'type': 'array', 'minItems': 1, 'maxItems': MAX_FACTS, 'items': fact},
        'dimensions': {'type': 'array', 'minItems': len(DIMENSIONS),
                       'maxItems': len(DIMENSIONS), 'items': {'$ref': '#/$defs/score'}}})
    schema = _object({'events': {'type': 'array', 'maxItems': MAX_EVENTS, 'items': item}})
    schema['$defs'] = {'frame_id': {'type': 'string', 'enum': frames}, 'score': score}
    return {'type': 'json_schema', 'json_schema': {
        'name': 'volleyball_event_anchors_v3', 'strict': True, 'schema': schema}}


def _named_dimensions(raw):
    """Map explicit dimension names without guessing from entry positions.

    A complete V2 named object remains losslessly compatible. Missing names,
    duplicates, aliases and extra fields cannot be converted into a valid score.
    """
    if isinstance(raw, list):
        if len(raw) != len(DIMENSIONS):
            raise ValueError('双榜维度不完整')
        dimensions = {}
        for item in raw:
            if not isinstance(item, dict) or set(item) != {'dimension', 'value', 'fact_indexes'}:
                raise ValueError('维度必须显式命名且引用事实索引')
            name = item['dimension']
            if not isinstance(name, str) or name not in DIMENSIONS or name in dimensions:
                raise ValueError('双榜维度名称重复或无效')
            dimensions[name] = {'value': item['value'], 'fact_indexes': item['fact_indexes']}
    else:
        dimensions = raw
    if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSIONS):
        raise ValueError('双榜维度不完整')
    return dimensions


def decode_event_response(data, evidence, start, end):
    """Compile wire references to exact observed PTS without inventing facts.

    Only clip bounds may grow to include the explicitly cited fact frames.
    Action bounds, peak, scores, confidence and completeness are never repaired.
    The original response is preserved by the caller for an auditable mapping.
    """
    from .events import number
    if not isinstance(data, dict) or set(data) != {'events'} or not isinstance(data['events'], list):
        raise ValueError('事件响应必须包含 events 数组')
    if len(data['events']) > MAX_EVENTS:
        raise ValueError('单块锚点事件过多')
    known = {e['id']: e for e in evidence}
    if len(known) != len(evidence):
        raise ValueError('输入证据 ID 重复')

    def frame_time(ref):
        raw = known.get(ref) if isinstance(ref, str) else None
        if not raw or raw['kind'] != 'frame' or not number(raw['start_sec'], start, end):
            raise ValueError('帧锚必须引用本上下文的画面证据')
        return raw['start_sec']

    events = []
    for wire in data['events']:
        if not isinstance(wire, dict) or set(wire) != set(EVENT_FIELDS+FRAME_FIELDS+('facts', 'dimensions')):
            raise ValueError('锚点事件字段不完整')
        times = {key.replace('_frame', '_sec'): frame_time(wire[key]) for key in FRAME_FIELDS}
        if not start <= times['clip_start_sec'] <= times['start_sec'] < times['end_sec'] <= times['clip_end_sec'] <= end:
            raise ValueError('事件帧锚顺序错误')
        if not times['start_sec'] <= times['peak_sec'] <= times['end_sec']:
            raise ValueError('回放峰值不属于事件')
        if not isinstance(wire['facts'], list) or not 1 <= len(wire['facts']) <= MAX_FACTS:
            raise ValueError('事件事实数量错误')
        event = {**{key: wire[key] for key in EVENT_FIELDS}, **times,
                 'observations': [], 'aftermath': [], 'reactions': [], 'dimensions': {}}
        facts = []
        for fact in wire['facts']:
            if not isinstance(fact, dict) or set(fact) != {'kind', 'text', 'frame_id', 'support_id'}:
                raise ValueError('锚点事实字段不完整')
            if not isinstance(fact['kind'], str) or fact['kind'] not in FACT_KINDS:
                raise ValueError('无效事实类别')
            if not isinstance(fact['text'], str) or not fact['text'].strip() or len(fact['text']) > 300:
                raise ValueError('无效事实描述')
            when = frame_time(fact['frame_id'])
            refs = [fact['frame_id']]
            support = fact['support_id']
            if support is not None:
                raw = known.get(support) if isinstance(support, str) else None
                if not raw or raw['kind'] not in ('audio', 'sound_event', 'local_motion'):
                    raise ValueError('事实辅助证据必须为本上下文的声音或运动测量')
                if not raw['start_sec']-1 <= when <= raw['end_sec']+1:
                    raise ValueError('事实帧时间与辅助证据不对应')
                refs.append(support)
            canonical = {'text': fact['text'], 'time_sec': when, 'evidence_ids': refs}
            facts.append(canonical)
            event[FACT_KINDS[fact['kind']]].append(canonical)
        event['clip_start_sec'] = min(event['clip_start_sec'], *(f['time_sec'] for f in facts))
        event['clip_end_sec'] = max(event['clip_end_sec'], *(f['time_sec'] for f in facts))
        dims = _named_dimensions(wire['dimensions'])
        for key, score in dims.items():
            if not isinstance(score, dict) or set(score) != {'value', 'fact_indexes'}:
                raise ValueError('维度必须引用事实索引')
            indexes = score['fact_indexes']
            if not isinstance(indexes, list) or len(indexes) > MAX_FACTS or any(
                    type(i) is not int or not 0 <= i < len(facts) for i in indexes):
                raise ValueError('评分引用了不存在的事实索引')
            if score['value'] is None:
                if indexes: raise ValueError('未知评分不能附带事实索引')
            elif not number(score['value'], 0, 4) or not indexes:
                raise ValueError('非未知评分必须引用事实且取值为 0–4')
            event['dimensions'][key] = {'value': score['value'],
                'evidence_ids': list(dict.fromkeys(ref for i in indexes for ref in facts[i]['evidence_ids']))}
        events.append(event)
    canonical = {'events': events}
    validate_events(canonical, evidence, start, end)
    return canonical


def event_response_format(evidence, start, end):
    """Constrain JSON shape, source-local times and available evidence IDs.

    JSON Schema cannot prove visual facts, causal links or ordering between
    fields. Every response must still pass events.validate_events afterwards.
    """
    def obj(properties):
        return {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}

    def string(limit):
        return {'type': 'string', 'minLength': 1, 'maxLength': limit}

    timestamp = {'type': 'number', 'minimum': start, 'maximum': end}
    ids = list(dict.fromkeys(item['id'] for item in evidence))
    if not ids:
        raise ValueError('事件理解缺少输入证据')
    refs = {'type': 'array', 'items': {'$ref': '#/$defs/evidence_id'}}
    fact = obj({'text': string(500), 'time_sec': timestamp,
                'evidence_ids': {**refs, 'minItems': 1}})
    score = obj({'value': {'anyOf': [{'type': 'number', 'minimum': 0, 'maximum': 4},
                                    {'type': 'null'}]}, 'evidence_ids': refs})
    unknown = obj({'value': {'type': 'null'}, 'evidence_ids': refs})
    dimensions = {key: {'$ref': '#/$defs/score'} for key in DIMENSIONS}
    if not any(e['kind'] in ('audio', 'sound_event') for e in evidence):
        dimensions['related_laughter'] = {'$ref': '#/$defs/unknown_score'}
    if not any(e['kind'] == 'local_motion' and e.get('measured') is True for e in evidence):
        dimensions['motion_intensity'] = {'$ref': '#/$defs/unknown_score'}
    event = obj({
        'event_type': string(80),
        **{key: timestamp for key in ('start_sec', 'end_sec', 'clip_start_sec', 'clip_end_sec', 'peak_sec')},
        'title': string(40), 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        **{key: {'type': ['boolean', 'null']} for key in
           ('is_rally', 'boundary_complete', 'injury_suspected', 'laughter_linked')},
        'uncertainty': string(800),
        **{key: {'type': 'array', 'items': {'$ref': '#/$defs/fact'},
                 **({'minItems': 1} if key == 'observations' else {})}
           for key in ('observations', 'aftermath', 'reactions')},
        'dimensions': obj(dimensions),
    })
    schema = obj({'events': {'type': 'array', 'maxItems': 30, 'items': event}})
    schema['$defs'] = {'evidence_id': {'type': 'string', 'enum': ids},
                       'fact': fact, 'score': score, 'unknown_score': unknown}
    return {'type': 'json_schema', 'json_schema': {
        'name': 'volleyball_events_v1', 'strict': True, 'schema': schema}}


def event_request_payload(model, prompt, content, evidence, start, end,
                          max_tokens=4096, reasoning_effort=None):
    """Shared by the production path and real-provider diagnostic probes."""
    payload = {'model': model, 'max_tokens': max_tokens,
        'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}],
        'response_format': wire_event_response_format(evidence, start, end)}
    if reasoning_effort is not None:
        payload['reasoning_effort'] = reasoning_effort
    return payload
