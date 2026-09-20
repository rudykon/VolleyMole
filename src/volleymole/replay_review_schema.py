"""Frame-grounded contracts for finding and checking a replay action sequence."""
import math

from .replay import ACTION_PRIORITY

ACTIONS = (*ACTION_PRIORITY, 'set', 'serve', 'other')


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


TEXT = {'type': 'string'}
FRAME = {'type': ['string', 'null']}
ACTION = {'type': 'string', 'enum': list(ACTIONS)}
SCORE = {'type': 'integer', 'minimum': 0, 'maximum': 5}
CONFIDENCE = {'type': 'number', 'minimum': 0, 'maximum': 1}
SCAN_SCHEMA = obj(dict(candidates={'type': 'array', 'maxItems': 4, 'items': obj(dict(
    action=ACTION, preparation_frame_id=TEXT, peak_frame_id=TEXT, result_frame_id=FRAME,
    excitement=SCORE, confidence=CONFIDENCE, observation=TEXT))}, uncertainty=TEXT))
REVIEW_SCHEMA = obj(dict(verdict={'type': 'string', 'enum': ['accept', 'expand', 'reject']},
    action=ACTION, lead_frame_id=FRAME, preparation_frame_id=FRAME, peak_frame_id=FRAME,
    result_frame_id=FRAME, tail_frame_id=FRAME, need_before={'type': 'boolean'},
    need_after={'type': 'boolean'}, start_complete={'type': 'boolean'}, end_complete={'type': 'boolean'},
    excitement=SCORE, confidence=CONFIDENCE, preparation_observation=TEXT,
    contact_observation=TEXT, result_observation=TEXT, reason=TEXT, uncertainty=TEXT))
CONTACTS = ('single_arm_attack', 'two_hand_set', 'forearm_defense', 'one_hand_save',
            'block_contact', 'missed_save', 'unclear')
OUTCOMES = ('opponent_response', 'team_return', 'ball_dead', 'unknown')
VERIFY_SCHEMA = obj({**REVIEW_SCHEMA['properties'],
    'contact_type': {'type': 'string', 'enum': list(CONTACTS)},
    'outcome_type': {'type': 'string', 'enum': list(OUTCOMES)},
    'contact_before_frame_id': FRAME, 'contact_after_frame_id': FRAME,
    'next_touch_frame_id': FRAME, 'outcome_before_frame_id': FRAME,
    'outcome_after_frame_id': FRAME, 'opening_in_motion': {'type': 'boolean'},
    'ending_in_motion': {'type': 'boolean'}})


def exact(data, schema):
    if not isinstance(data, dict) or set(data) != set(schema['properties']):
        raise ValueError('replay_response_fields')


def text(value, nonempty=False):
    if not isinstance(value, str) or len(value) > 1200 or (nonempty and not value.strip()):
        raise ValueError('replay_response_text')


def scores(row):
    if row['action'] not in ACTIONS:
        raise ValueError('replay_action')
    if type(row['excitement']) is not int or not 0 <= row['excitement'] <= 5:
        raise ValueError('replay_excitement')
    if type(row['confidence']) not in (int, float) or not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1:
        raise ValueError('replay_confidence')


def frame_times(evidence):
    refs = {r['id']: r['start_sec'] for r in evidence if r['kind'] == 'frame'}
    if not refs or len(refs) != len(evidence) or not all(type(t) in (int, float) and math.isfinite(t) for t in refs.values()):
        raise ValueError('replay_frame_evidence')
    return refs


def anchor(row, key, refs, nullable=False):
    value = row[key]
    if nullable and value is None:
        return None
    if not isinstance(value, str) or value not in refs:
        raise ValueError('replay_unknown_frame')
    return refs[value]


def decode_scan(data, evidence):
    exact(data, SCAN_SCHEMA); text(data['uncertainty'])
    if not isinstance(data['candidates'], list) or len(data['candidates']) > 4:
        raise ValueError('replay_candidate_count')
    refs = frame_times(evidence)
    rows = []
    for row in data['candidates']:
        exact(row, SCAN_SCHEMA['properties']['candidates']['items']); scores(row)
        text(row['observation'], True)
        start = anchor(row, 'preparation_frame_id', refs)
        peak = anchor(row, 'peak_frame_id', refs)
        end = anchor(row, 'result_frame_id', refs, True)
        if not start < peak or (end is not None and not peak < end):
            raise ValueError('replay_candidate_order')
        rows.append({**row, 'action_start_sec': start, 'peak_sec': peak, 'action_end_sec': end})
    return rows


def decode_review(data, evidence):
    exact(data, REVIEW_SCHEMA); scores(data)
    for k in ('preparation_observation', 'contact_observation', 'result_observation', 'reason', 'uncertainty'):
        text(data[k], k == 'reason' or (data['verdict'] == 'accept' and k != 'uncertainty'))
    if data['verdict'] not in ('accept', 'expand', 'reject'):
        raise ValueError('replay_verdict')
    for k in ('need_before', 'need_after', 'start_complete', 'end_complete'):
        if type(data[k]) is not bool:
            raise ValueError('replay_boundary_boolean')
    refs = frame_times(evidence)
    names = ('lead', 'preparation', 'peak', 'result', 'tail')
    times = {k: anchor(data, k+'_frame_id', refs, data['verdict'] != 'accept') for k in names}
    if data['verdict'] == 'accept':
        a, start, peak, end, b = (times[k] for k in names)
        if not a <= start < peak < end <= b:
            raise ValueError('replay_review_order')
        if data['need_before'] or data['need_after'] or not (data['start_complete'] and data['end_complete']):
            raise ValueError('replay_contradictory_accept')
    if data['verdict'] == 'expand' and not (data['need_before'] or data['need_after']):
        raise ValueError('replay_expansion_direction')
    return {**data, 'times': times}


def decode_verify(data, evidence):
    """A separate observation, with visible bracketing frames for two events.

    A valid JSON answer can still reject a proposed action. Do not treat such a
    semantic disagreement as a transport error or retry until it says yes.
    """
    exact(data, VERIFY_SCHEMA)
    base = decode_review({k:data[k] for k in REVIEW_SCHEMA['properties']}, evidence)
    if data['contact_type'] not in CONTACTS or data['outcome_type'] not in OUTCOMES:
        raise ValueError('replay_verify_event_type')
    for k in ('opening_in_motion', 'ending_in_motion'):
        if type(data[k]) is not bool:
            raise ValueError('replay_verify_motion_boolean')
    refs = frame_times(evidence)
    names = ('contact_before', 'contact_after', 'next_touch', 'outcome_before', 'outcome_after')
    events = {k:anchor(data, k+'_frame_id', refs, True) for k in names}
    if data['verdict'] == 'accept':
        if any(events[k] is None for k in names if k != 'next_touch'):
            raise ValueError('replay_verify_missing_event_frames')
        t = base['times']
        if not (t['lead'] <= events['contact_before'] < t['peak'] < events['contact_after']
                <= t['result'] and t['peak'] <= events['outcome_before'] < t['result']
                < events['outcome_after'] <= t['tail']):
            raise ValueError('replay_verify_event_order')
        if events['next_touch'] is not None and not t['peak'] < events['next_touch'] <= t['result']:
            raise ValueError('replay_verify_next_touch_order')
    return {**base, **{k:data[k] for k in VERIFY_SCHEMA['properties'] if k not in REVIEW_SCHEMA['properties']},
            'event_times': events}


def payload(model, phase, prompt, content, evidence, max_tokens, reasoning_effort=None):
    import copy
    import json
    schema = copy.deepcopy({'scan':SCAN_SCHEMA, 'review':REVIEW_SCHEMA, 'verify':VERIFY_SCHEMA}[phase])
    ids = [r['id'] for r in evidence]
    target = schema['properties']['candidates']['items']['properties'] if phase == 'scan' else schema['properties']
    for k, v in target.items():
        if k.endswith('_frame_id'):
            v['enum'] = ids + ([None] if isinstance(v['type'], list) else [])
    # Some compatible gateways return JSON without enforcing or exposing the
    # response_format schema. Describe the same contract to the model itself;
    # still validate the unmodified response locally, never coerce bad anchors.
    prompt += ('\n\n输出必须严格符合以下 JSON Schema。字段名使用 action，不是 action_type；'
               '帧号必须是输入中完整的字符串（例如 frame_00003），不能输出数字序号。'
               '不得增加、遗漏或重命名字段。\n' + json.dumps(schema, ensure_ascii=False))
    result = dict(model=model, temperature=0, max_tokens=max_tokens,
        messages=[dict(role='system', content=prompt), dict(role='user', content=content)],
        response_format=dict(type='json_schema', json_schema=dict(name='replay_'+phase, strict=True, schema=schema)))
    if reasoning_effort is not None:
        result['reasoning_effort'] = reasoning_effort
    return result
