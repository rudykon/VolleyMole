"""Automatic scan -> dense sequence review -> bounded context expansion.

Only source-bound observations become replay windows. Ranking previews and the
previous 9.15 review are never inputs. Each request and complete rally result is
cached independently, so a failed request can resume without losing other work.
"""
import hashlib
import json
from pathlib import Path
import time

from .common import APP, digest, read_json, save_json
from .replay import ACTION_PRIORITY, reviewed_window
from .replay_review_schema import decode_scan, decode_review, decode_verify
from .semantic import sampled_evidence, request_json
from .replay_motion import boundary_guard, spatial_hints

VERSION = 2


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def implementation():
    files = ('replay_review.py', 'replay_review_schema.py', 'replay_evidence.py', 'replay_requests.py', 'replay_motion.py', 'replay.py', 'semantic.py', 'llm_transport.py',
             'prompts/replay_scan.md', 'prompts/replay_review.md', 'prompts/replay_verify.md')
    return {p: digest(APP/p) for p in files}


def public_settings(settings):
    # An allowlist keeps credentials out of signatures, prompts and reports.
    return {k: settings[k] for k in ('endpoint', 'model', 'timeout', 'max_tokens', 'reasoning_effort',
        'scan_fps', 'review_fps', 'verify_fps', 'width', 'chunk_sec', 'overlap_sec', 'max_context_sec',
        'max_expansions', 'max_candidates', 'min_confidence', 'min_excitement', 'retries',
        'overview_frames', 'detail_limit') if k in settings}


def scan_windows(start, end, size, overlap):
    if not (0 <= start < end and size > overlap >= 0):
        raise ValueError('replay_scan_window')
    windows = []
    while start < end-1e-8:
        right = min(end, start+size)
        windows.append((start, right))
        if right == end:
            break
        start = right-overlap
    return windows


DECODERS = dict(scan=decode_scan, review=decode_review, verify=decode_verify)


def request_phase(source, job, settings, cache, deadline, code=None):
    from .replay_requests import request_phase as request
    return request(source, job, settings, cache, deadline,
                   decoder=DECODERS[job['phase']], request_fn=request_json, sample_fn=sampled_evidence)


def distinct_candidates(rows, settings):
    # A coarse scan nominates hypotheses, not final approvals. Low confidence
    # about a distant contact is a reason to inspect dense frames, not to skip
    # that inspection. The final review still uses min_confidence unchanged.
    eligible = [r for r in rows if r['action'] in ACTION_PRIORITY
                and r['confidence'] >= min(.4, settings['min_confidence'])
                and r['excitement'] >= settings['min_excitement']]
    # Watchability before action class: a difficult save can beat an ordinary spike.
    ordered = sorted(eligible, key=lambda r: (r['excitement'], r['confidence'], ACTION_PRIORITY[r['action']]), reverse=True)
    unique = []
    for r in ordered:
        if not any(abs(r['peak_sec']-v['peak_sec']) < .8 for v in unique):
            unique.append(r)
    # Reserve one place for a different action family: coarse confidence must
    # not erase a difficult save in a long rally containing ordinary attacks.
    if len(unique) > 1 and settings['max_candidates'] > 1:
        other = next((r for r in unique[1:] if (r['action'] in ('dig','receive')) !=
                      (unique[0]['action'] in ('dig','receive'))), None)
        if other is not None:
            unique.remove(other); unique.insert(1, other)
    return unique[:settings['max_candidates']]


def initial_context(candidate, item, settings):
    a = max(item['clip_start_sec'], min(candidate['action_start_sec']-.75, candidate['peak_sec']-2.))
    b = min(item['clip_end_sec'], max((candidate['action_end_sec'] or candidate['peak_sec']+3.)+1., candidate['peak_sec']+3.))
    if b-a > settings['max_context_sec']:
        return None
    return a, b


def expand_context(a, b, before, after, item, settings):
    # Grow enough to expose new action context; never step into another selected
    # rally or crop a different side to make the frame budget fit.
    step = max(1.5, (b-a)*.4)
    left = max(item['clip_start_sec'], a-step) if before else a
    right = min(item['clip_end_sec'], b+step) if after else b
    if ((before and left >= a-1e-6) or (after and right <= b+1e-6)
            or right-left > settings['max_context_sec']):
        return None
    return left, right


def omission(source, reason, **extra):
    return dict(status='omit', method='vision_sequence_review', source_sha256=source['identity']['sha256'],
                reason=reason, **extra)



def cut_times(result):
    """Add editing handles using already inspected source frames.

    The model identifies the events. Numeric handle requirements are compiled
    locally instead of repeatedly asking it to count fractions of a second.
    Extending either side never removes any of the approved observed action.
    """
    t = dict(result['times'])
    frames = result.get('_observed_frame_times', [])
    left = min(t['lead'], t['preparation']-.5, t['peak']-1.25)
    right = max(t['tail'], t['result']+.5, t['peak']+1.6)
    before = [v for v in frames if v <= left+1e-6]
    after = [v for v in frames if v >= right-1e-6]
    if before:
        t['lead'] = max(before)
    if after:
        t['tail'] = min(after)
    return t


def window_from_verified(result, item, source):
    from fractions import Fraction
    t = cut_times(result)
    return dict(source_start_sec=t['lead'], action_start_sec=t['preparation'], peak_sec=t['peak'],
                action_end_sec=t['result'], source_end_sec=min(item['clip_end_sec'],t['tail']+1/float(Fraction(source.get('nominal_fps','30/1')))),
                action=result['action'], confidence=result['confidence'], excitement=result['excitement'])


def confidence_problem(result, settings):
    if result['confidence'] < settings['min_confidence']:
        return dict(unresolved=True, reason='insufficient_visual_confidence')
    return None


def contact_family_problem(draft, verified):
    """Conflicting descriptions of one contact need independent resolution.

    Compare the original draft contact, not the current magnification focus:
    a later, refocused contact may legitimately have a different action family.
    This finding stays local and is never added to the blind verifier's input.
    """
    defense = {'dig', 'receive'}
    changed = ((draft['action'] in defense and verified['action'] == 'spike')
               or (draft['action'] == 'spike' and verified['action'] in defense))
    if changed and abs(draft['times']['peak']-verified['times']['peak']) <= .5:
        return dict(unresolved=True, reason='conflicting_action_families_for_same_contact')
    return None


def verification_problem(result, settings):
    """Separate uncertain visual evidence from a supported semantic rejection."""
    uncertain = confidence_problem(result, settings)
    if uncertain:
        return uncertain
    allowed = {'spike':{'single_arm_attack'},'block':{'block_contact'},
               'dig':{'forearm_defense','one_hand_save','missed_save'}, 'receive':{'forearm_defense'}}
    if (result['contact_type'] not in allowed.get(result['action'], set())
            or result['excitement'] < settings['min_excitement']):
        return dict(reject=True, reason='unverified_action_or_insufficient_visual_evidence')
    t,e = cut_times(result),result['event_times']
    fps = settings.get('verify_fps',settings['review_fps'])
    if max(t['peak']-e['contact_before'],e['contact_after']-t['peak']) > max(.5,3/fps):
        return dict(reject=True,reason='contact_not_resolved_in_neighboring_frames')
    before = (result['opening_in_motion'] or t['preparation']-t['lead'] < .35
              or t['peak']-t['preparation'] < .2 or t['peak']-t['lead'] < 1.)
    after = (result['ending_in_motion'] or result['outcome_type']=='unknown'
             or (result['action']=='spike' and result['outcome_type']=='team_return')
             or (result['contact_type']=='missed_save' and result['outcome_type']!='ball_dead')
             or (result['outcome_type']!='ball_dead' and e['next_touch'] is None)
             or t['tail']-t['result'] < .35 or t['tail']-t['peak'] < 1.5)
    if before or after:
        return dict(need_before=before,need_after=after,reason='unresolved_preparation_or_observed_outcome',
                    previous_anchors=t,instruction='根据真实启动前/后续处理后的画面重新选择端点，不能把球的飞行顶点当作触球或结果。')
    return None

def review_rally(item, source, settings, cache, deadline, code=None, *, motion=None):
    code = code or implementation()
    identity = dict(version=VERSION, source=source['identity'],
        source_clock={k: source.get(k) for k in ('start_sec', 'rotation', 'nominal_fps', 'duration_sec')},
        interval=[item['clip_start_sec'], item['clip_end_sec']], settings=public_settings(settings), implementation=code,
        motion_sha256=fingerprint(motion) if motion else None)
    signature = fingerprint(identity); target = Path(cache)/'rallies'/f'{signature}.json'
    if target.is_file():
        try:
            saved = read_json(target)
            if saved['identity'] == identity and saved['signature'] == signature and saved['status'] == 'complete':
                decoded_requests = {}
                for request in saved['requests']:
                    if digest(request['path']) != request['sha256']:
                        raise ValueError('replay_cache_changed')
                    raw = read_json(request['path'])
                    decoded = DECODERS[request['phase']](raw['raw'], raw['evidence'])
                    if request['phase']=='verify':
                        decoded['_observed_frame_times'] = [r['start_sec'] for r in raw['evidence']]
                    decoded_requests[request['signature']] = (request['phase'], decoded)
                if saved['review']['status']=='approved':
                    phase, decoded = decoded_requests[saved['review']['evidence_request']]
                    if phase != 'verify' or decoded['verdict'] != 'accept':
                        raise ValueError('replay_cache_verdict')
                    expected = window_from_verified(decoded, item, source)
                    if any(saved['review'].get(k) != v for k,v in expected.items()):
                        raise ValueError('replay_cache_not_grounded')
                    if verification_problem(decoded, settings):
                        raise ValueError('replay_cache_quality_gate')
                    if motion and boundary_guard(motion, saved['review'])['verdict'] == 'expand':
                        raise ValueError('replay_cache_motion_conflict')
                reviewed_window(item, {'replay_review': saved['review']}, source)
                return {**saved, 'cached': True, 'artifact': str(target)}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    requests = []; candidates = []; attempts = []
    def call(job):
        data, audit = request_phase(source, job, settings, cache, deadline, code)
        if job['phase']=='verify':
            data['_observed_frame_times'] = [r['start_sec'] for r in audit['evidence']]
        requests.append(dict(phase=job['phase'], context=[job['start'], job['end']],
            path=audit['artifact'], sha256=digest(audit['artifact']), cached=audit['cached'],
            frame_count=len(audit['evidence']), signature=audit['signature']))
        return data
    for index, (a, b) in enumerate(scan_windows(item['clip_start_sec'], item['clip_end_sec'], settings['chunk_sec'], settings['overlap_sec'])):
        candidates.extend(call(dict(phase='scan', index=index, start=a, end=b)))
    shortlisted = distinct_candidates(candidates, settings)
    approved = []; unresolved = []
    for candidate in shortlisted:
        context = initial_context(candidate, item, settings)
        if context is None:
            unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], reason='context_exceeds_budget'))
            continue
        problem = None; context_expansions = 0; anchor_rechecks = 0
        draft = None
        for expansion in range(settings['max_expansions']+1):
            a,b = context
            job = dict(phase='review', start=a, end=b, candidate=candidate, boundary_problem=problem)
            if motion:
                job['spatial_hints'] = spatial_hints(motion, start_sec=max(a,candidate['peak_sec']-1),
                                                     end_sec=min(b,candidate['peak_sec']+1))
            result = call(job)
            attempt = dict(phase='review', candidate_peak_sec=candidate['peak_sec'],
                           scan_peak_sec=candidate['peak_sec'], context=[a,b], expansion=expansion, result=result)
            attempts.append(attempt)
            if result['verdict'] == 'reject':
                uncertain = confidence_problem(result, settings)
                if uncertain:
                    attempt['quality_problem'] = uncertain
                    unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], phase='review',
                                           reason=uncertain['reason'], problem=uncertain))
                break
            if result['verdict'] == 'accept':
                draft = result
                break
            if expansion == settings['max_expansions']:
                unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], reason='expansion_limit'))
                break
            problem = dict(need_before=result['need_before'], need_after=result['need_after'], previous_reason=result['reason'])
            context = expand_context(a,b,result['need_before'],result['need_after'],item,settings)
            if context is None:
                unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], reason='context_or_selected_clip_limit'))
                break
            context_expansions += 1
        if draft is None:
            continue
        # The verifier receives only a source time, never the draft class,
        # excitement, observations, or proposed boundaries. A longer context
        # exposes continued ball flight after a falsely accepted draft ending.
        focus = draft['times']['peak']
        context = (max(item['clip_start_sec'], min(draft['times']['lead']-1,focus-3)),
                   min(item['clip_end_sec'], max(draft['times']['tail']+2,focus+6)))
        if context[1]-context[0] > settings['max_context_sec']:
            unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], focus_sec=focus,
                                   reason='verification_context_exceeds_budget'))
            continue
        problem = None
        for expansion in range(settings['max_expansions']+1):
            a,b = context
            job = dict(phase='verify', start=a, end=b, focus_sec=focus, boundary_problem=problem)
            if motion:
                job['spatial_hints'] = spatial_hints(motion, start_sec=max(a,focus-.5), end_sec=min(b,focus+.5))
            result = call(job)
            attempt = dict(phase='verify', candidate_peak_sec=focus, scan_peak_sec=candidate['peak_sec'],
                           context=[a,b], expansion=expansion, result=result)
            attempts.append(attempt)
            if result['verdict'] == 'reject':
                uncertain = confidence_problem(result, settings)
                if uncertain:
                    attempt['quality_problem'] = uncertain
                    unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], phase='verify',
                                           reason=uncertain['reason'], problem=uncertain))
                break
            before,after = result['need_before'],result['need_after']
            if result['verdict'] == 'accept':
                uncertain = confidence_problem(result, settings)
                if uncertain:
                    attempt['quality_problem'] = uncertain
                    unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], phase='verify',
                                           reason=uncertain['reason'], problem=uncertain))
                    break
                if abs(result['times']['peak']-focus) > .5:
                    # A corrected contact needs its own consecutive native
                    # detail frames. Do not approve a new distant contact from
                    # a window magnified around the previous hypothesis.
                    attempt['quality_problem'] = dict(reason='contact_requires_refocused_detail')
                    if expansion == settings['max_expansions']:
                        unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], focus_sec=focus,
                                               reason='contact_refocus_limit'))
                        break
                    focus = result['times']['peak']
                    context = (max(item['clip_start_sec'],focus-3),min(item['clip_end_sec'],focus+6))
                    problem = None
                    continue
                problem = verification_problem(result, settings)
                if problem and problem.get('reject'):
                    attempt['quality_problem'] = problem
                    break
                conflict = contact_family_problem(draft, result)
                if conflict:
                    attempt['quality_problem'] = conflict
                    unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], phase='verify',
                                           reason=conflict['reason'], problem=conflict))
                    break
                before,after = (problem or {}).get('need_before',False),(problem or {}).get('need_after',False)
                selected_window = window_from_verified(result,item,source)
                guard = boundary_guard(motion or {}, selected_window)
                attempt['motion_guard'] = guard
                if guard['verdict'] == 'expand':
                    after = True
                    problem = dict(need_before=before,need_after=True, reason='observed_flight_continues_before_next_touch',
                                   minimum_source_end_sec=guard['minimum_source_end_sec'])
                if not problem:
                    approved.append(dict(status='approved', method='vision_sequence_review', boundary_complete=True,
                        source_sha256=source['identity']['sha256'], **selected_window,
                        reason=result['reason'], uncertainty=result['uncertainty'],
                        observations={k:result[k] for k in ('preparation_observation','contact_observation','result_observation')},
                        model=settings['model'], evidence_request=requests[-1]['signature'],
                        review_signature=signature, boundary_expansions=context_expansions, anchor_rechecks=anchor_rechecks,
                        independent_verification={k:result[k] for k in ('contact_type','outcome_type','event_times','opening_in_motion','ending_in_motion')},
                        motion_guard=guard))
                    break
            else:
                problem = dict(need_before=before,need_after=after, reason=result['reason'])
            attempt['quality_problem'] = problem
            if expansion == settings['max_expansions']:
                unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], focus_sec=focus,
                                       context=[a,b], reason='verification_incomplete', problem=problem))
                break
            expanded = expand_context(a,b,before,after,item,settings)
            if expanded:
                context = expanded; context_expansions += 1
            elif result['verdict']=='accept':
                # Existing frames may already cover better anchors; ask once
                # more without pretending the clip edges can be extended.
                anchor_rechecks += 1
            else:
                unresolved.append(dict(candidate_peak_sec=candidate['peak_sec'], focus_sec=focus,
                                       context=[a,b], reason='context_or_selected_clip_limit'))
                break
    # Every shortlisted candidate is assessed. The first accepted jump is not
    # automatically the most watchable action in a multi-exchange rally.
    selected = max(approved, key=lambda r:(r['excitement'], r['confidence'],
                   r['action'] in ('dig','block'), r['action_end_sec']-r['peak_sec'])) if approved else None
    provisional = selected
    for pending in unresolved:
        candidate = next(c for c in shortlisted if c['peak_sec']==pending['candidate_peak_sec'])
        pending['scan_excitement'] = candidate['excitement']
        pending['scan_action'] = candidate['action']
        # Dense observations can reveal a harder save or correct a coarse
        # action label. Retain every such hypothesis as a reason to withhold a
        # weaker selection, never as evidence that approves the uncertain play.
        hypotheses = [candidate]+[a['result'] for a in attempts
                                  if a['scan_peak_sec']==candidate['peak_sec']]
        pending['potential_excitement'] = max(h['excitement'] for h in hypotheses)
        pending['potential_actions'] = sorted({h['action'] for h in hypotheses})
    if selected and any(p['potential_excitement'] > selected['excitement'] or
        (p['potential_excitement'] == selected['excitement']
         and set(p['potential_actions']) & {'dig','receive'} and selected['action']=='spike')
        for p in unresolved):
        selected = None
    if selected is None:
        reason = ('候选动作的类别、触球或完整边界仍有未解决的不确定，不能确认当前回放是最佳选择。' if unresolved else
                  '通看回合后没有足够证据支持值得慢放的进攻或救球。' if not shortlisted else
                  '候选动作在加密复核中被否决，不生成慢回放。')
        selected = omission(source, reason, review_signature=signature, model=settings['model'])
    status = 'incomplete' if selected['status']=='omit' and unresolved else 'complete'
    saved = dict(signature=signature, identity=identity, status=status, review=selected,
        candidates=candidates, shortlisted=shortlisted, verified_candidates=approved, provisional_review=provisional,
        attempts=attempts, requests=requests,
        unresolved_boundaries=unresolved)
    save_json(target, saved)
    return {**saved, 'cached': False, 'artifact': str(target)}
