"""Independent observation for a proposed heavy spike, without meme priming."""
from .meme_director import response_schema


def observation_schema():
    properties = {key: {'type': 'string'} for key in (
        'target_player', 'observed_motion', 'ball_trajectory_before',
        'ball_trajectory_after', 'visible_contact_description')}
    properties.update(
        action={'type': 'string', 'enum': response_schema()['properties']['action']['enum']},
        force={'type': 'string', 'enum': ['visibly_hard', 'soft', 'not_determinable']})
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def observation_payload(prepared, job, directory):
    from .meme_stage import payload
    wire = payload(prepared, job, directory)
    peak = job['clip']['peak_sec']
    selected = sorted({min(range(len(job['evidence'])), key=lambda i: abs(
        job['evidence'][i]['start_sec'] - peak - delta))
        for delta in [-.5, -.3, -.15, 0, .15, .3, .5]})
    content = [{'type': 'text', 'text': (
        f'Follow the player handling the ball around source_sec={peak}. '
        'Describe only visible changes in these timestamped images. FULL and SAME FRAME DETAIL '
        'repeat the SAME instant. Do not infer a strike merely from a raised arm. '
        'Identify the ball trajectory before and after the contact.')}]
    for index in selected:
        content.extend(wire['messages'][1]['content'][1+index*2:3+index*2])
    return {'model': wire['model'], 'max_tokens': 8192, 'reasoning_effort': 'high',
        'messages': [{'role': 'system', 'content': (
            'You inspect visible volleyball motion. Report observations, not excitement or sound effects. '
            'If a fact is not visible, say unknown. Return JSON.')},
            {'role': 'user', 'content': content}],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'action_observation', 'strict': True, 'schema': observation_schema()}}}


def validate_observation(raw):
    properties = observation_schema()['properties']
    if not isinstance(raw, dict) or set(raw) != set(properties):
        raise ValueError('meme_observation_fields')
    for name, spec in properties.items():
        value = raw[name]
        if (not isinstance(value, str) or not 1 <= len(value.strip()) <= 2000
                or ('enum' in spec and value not in spec['enum'])):
            raise ValueError('meme_observation_value')
    return raw


def confirms_heavy_spike(raw):
    raw = validate_observation(raw)
    return raw['action'] == 'power_spike' and raw['force'] == 'visibly_hard'
