"""Evidence-linked meme decisions and conservative compilation to audio cues."""
import math
from pathlib import Path

from .common import APP, read_json


def policy():
    return read_json(APP / 'meme_policy.json')


def response_schema(rules=None):
    rules = policy() if rules is None else rules
    tags = sorted(set(rules['global_forbidden']).union(*(
        set(r['require_all'] + r['require_any'] + r['forbid']) for r in rules['rules'])))
    properties = {
        'decision': {'type': 'string', 'enum': ['use', 'skip']},
        'meme_id': {'type': ['string', 'null'], 'enum': [r['id'] for r in rules['rules']] + [None]},
        'confidence': {'type': 'number'},
        'action': {'type': 'string', 'enum': ['power_spike', 'tip', 'roll_shot', 'set', 'serve', 'dig', 'block', 'rally', 'other', 'unknown']},
        'power': {'type': 'string', 'enum': ['hard', 'soft', 'unknown', 'not_applicable']},
        'facts': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
            'properties': {'tag': {'type': 'string', 'enum': tags}, 'frame_ids': {'type': 'array', 'items': {'type': 'string'}},
                           'description': {'type': 'string'}}, 'required': ['tag', 'frame_ids', 'description']}},
        'anchor_frame_id': {'type': ['string', 'null']},
        'placement': {'type': ['string', 'null'], 'enum': ['before_action', 'after_action', None]},
        'contact_frame_ids': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'},
    }
    return {'type': 'object', 'additionalProperties': False, 'properties': properties, 'required': list(properties)}


def validate_decision(raw, evidence, rules):
    schema = response_schema()
    if not isinstance(raw, dict) or set(raw) != set(schema['properties']):
        raise ValueError('meme_response_fields')
    for name in ('decision', 'action', 'power', 'placement'):
        if raw[name] not in schema['properties'][name]['enum']:
            raise ValueError('meme_response_enum')
    confidence = raw['confidence']
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('meme_response_confidence')
    if not isinstance(raw['reason'], str) or not 1 <= len(raw['reason'].strip()) <= 1000:
        raise ValueError('meme_response_reason')
    frames = {r['id']: r['start_sec'] for r in evidence}
    def references(values, minimum):
        if (not isinstance(values, list) or not minimum <= len(values) <= 64
                or any(not isinstance(v, str) or v not in frames for v in values)
                or len(set(values)) != len(values)
                or any(frames[a] >= frames[b] for a, b in zip(values, values[1:]))):
            raise ValueError('meme_unknown_or_unordered_frames')
    references(raw['contact_frame_ids'], 0)
    tags = set(rules['global_forbidden'])
    for rule in rules['rules']:
        for field in ('require_all', 'require_any', 'forbid'):
            tags.update(rule[field])
    if not isinstance(raw['facts'], list) or len(raw['facts']) > 32:
        raise ValueError('meme_response_facts')
    seen = set()
    for fact in raw['facts']:
        if (not isinstance(fact, dict) or set(fact) != {'tag', 'frame_ids', 'description'}
                or not isinstance(fact['tag'], str) or fact['tag'] not in tags or fact['tag'] in seen
                or not isinstance(fact['description'], str) or not 1 <= len(fact['description'].strip()) <= 500):
            raise ValueError('meme_response_fact')
        references(fact['frame_ids'], 1)
        seen.add(fact['tag'])
    if raw['decision'] == 'skip':
        if any(raw[k] is not None for k in ('meme_id', 'anchor_frame_id', 'placement')):
            raise ValueError('meme_contradictory_skip')
    else:
        if not isinstance(raw['meme_id'], str) or raw['meme_id'] not in {r['id'] for r in rules['rules']}:
            raise ValueError('meme_unknown_id')
        references([raw['anchor_frame_id']], 1)
        if raw['placement'] is None or not raw['contact_frame_ids']:
            raise ValueError('meme_missing_timing')
    return raw


def eligible_cue(raw, evidence, clip, assets, rules):
    """Local vetoes never upgrade a skip, invent facts, or substitute another meme."""
    validate_decision(raw, evidence, rules)
    if raw['decision'] == 'skip':
        return None, 'model_skip'
    rule = next(r for r in rules['rules'] if r['id'] == raw['meme_id'])
    facts = {f['tag']: f for f in raw['facts']}
    tags = set(facts)
    correction = clip.get('editor_constraints', {})
    if raw['meme_id'] in correction.get('forbidden_memes', []):
        return None, 'editor_veto'
    if tags.intersection(rules['global_forbidden'] + rule['forbid']):
        return None, 'forbidden_condition'
    if raw['confidence'] < rule['min_confidence']:
        return None, 'insufficient_confidence'
    if not set(rule['require_all']) <= tags or (rule['require_any'] and not tags.intersection(rule['require_any'])):
        return None, 'missing_required_evidence'
    if rule.get('requires_story_context') and not clip.get('story_context'):
        return None, 'missing_story_context'
    frames = {r['id']: r['start_sec'] for r in evidence}
    if rule['id'] == 'kaipao':
        if (raw['action'] != 'power_spike' or raw['power'] != 'hard'
                or correction.get('action') in ('tip', 'roll_shot', 'set', 'serve', 'dig', 'block')):
            return None, 'not_confirmed_heavy_spike'
        flight = [frames[f] for f in facts['fast_post_contact_flight']['frame_ids']]
        contact = [frames[f] for f in facts['forceful_contact']['frame_ids']]
        swing = [frames[f] for f in facts['full_arm_swing']['frame_ids']]
        if len(flight) < 3 or flight[-1] - flight[0] < .12 or flight[0] < min(contact) or min(swing) >= max(contact):
            return None, 'missing_ordered_power_evidence'
    asset = assets.get(rule['id'])
    if not asset or asset.get('ready') is not True or asset.get('kind') != 'audio':
        return None, 'asset_unavailable'
    if rule.get('requires_censored_asset') and asset.get('censored') is not True:
        return None, 'censored_asset_required'
    if raw['placement'] not in rule['placements']:
        return None, 'placement_not_allowed'
    anchor = frames[raw['anchor_frame_id']]
    if not clip['source_start_sec'] <= anchor < clip['source_end_sec']:
        return None, 'anchor_outside_replay'
    rate = clip['playback_rate']
    mapped = lambda t: clip['output_start_sec'] + (t - clip['source_start_sec']) / rate
    start = mapped(anchor)
    end = start + asset['source_end_sec'] - asset['source_start_sec']
    if end > clip['output_end_sec'] + 1e-6:
        return None, 'phrase_would_be_truncated'
    contacts = [mapped(frames[f]) for f in raw['contact_frame_ids']]
    contacts.append(mapped(clip['peak_sec']))
    if any(start < t + .15 and end > t - .15 for t in contacts):
        return None, 'would_cover_ball_contact'
    primary = mapped(clip['peak_sec'])
    if ((raw['placement'] == 'before_action' and end > primary - .15)
            or (raw['placement'] == 'after_action' and start < primary + .15)):
        return None, 'wrong_side_of_action'
    # Outcome-based praise cannot precede any of the model's cited outcome evidence.
    outcome_tags = ('successful_outcome', 'outcome_visible', 'visible_reversal', 'comic_reaction')
    observed = [frames[f] for tag in outcome_tags if tag in facts for f in facts[tag]['frame_ids']]
    if raw['placement'] == 'after_action' and observed and anchor < max(observed):
        return None, 'outcome_not_yet_visible'
    return {'id': f'{clip["id"]}_{rule["id"]}', 'meme_id': rule['id'], 'label': rule['label'],
            'rally_id': clip['id'], 'asset': asset['path'], 'at_sec': start,
            'source_start_sec': asset['source_start_sec'], 'source_end_sec': asset['source_end_sec'],
            'rms_db': asset.get('rms_db', -20), 'duck_db': -3,
            'confidence': raw['confidence'], 'reason': raw['reason'],
            'anchor_frame_id': raw['anchor_frame_id'], 'facts': raw['facts']}, 'eligible'


def compile_plan(jobs, results, assets, rules, *, observations=None):
    from .meme_verifier import confirms_heavy_spike
    observations = observations or {}
    candidates, audit = [], []
    for job in jobs:
        rid = job['clip']['id']
        if rid not in results:
            audit.append({'rally_id': rid, 'status': 'unreviewed'})
            continue
        cue, reason = eligible_cue(results[rid], job['evidence'], job['clip'], assets, rules)
        if cue and cue['meme_id'] == 'kaipao':
            if rid not in observations:
                cue, reason = None, 'independent_action_review_required'
            elif not confirms_heavy_spike(observations[rid]):
                cue, reason = None, 'independent_action_disagrees'
        audit.append({'rally_id': rid, 'status': reason})
        if cue:
            candidates.append(cue)
    density = rules['density']
    selected = []
    for cue in sorted(candidates, key=lambda c: (-c['confidence'], c['at_sec'])):
        rejected = (len(selected) >= density['max_cues']
            or any(c['rally_id'] == cue['rally_id'] for c in selected)
            or (not density['repeat_meme'] and any(c['meme_id'] == cue['meme_id'] for c in selected))
            or any(max(c['at_sec'], cue['at_sec']) - min(
                c['at_sec'] + c['source_end_sec'] - c['source_start_sec'],
                cue['at_sec'] + cue['source_end_sec'] - cue['source_start_sec']) < density['min_gap_sec'] for c in selected))
        next(row for row in audit if row['rally_id'] == cue['rally_id'])['status'] = 'density_skip' if rejected else 'selected'
        if not rejected:
            selected.append(cue)
    return {'version': 1, 'max_cues': density['max_cues'], 'selection_mode': 'vision_api_with_policy_vetoes',
            'cues': sorted(selected, key=lambda c: c['at_sec']), 'decisions': audit,
            'review_complete': len(results) == len(jobs) and not any(
                row['status'] == 'independent_action_review_required' for row in audit)}
