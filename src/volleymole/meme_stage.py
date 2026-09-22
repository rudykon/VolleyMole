"""Prepare local evidence, then explicitly run a bounded API meme review."""
import argparse
import base64
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

from .common import APP, digest, probe, read_json, save_json
from .meme_audio import finite_number, media_info
from .meme_director import compile_plan, eligible_cue, policy, response_schema, validate_decision
from .replay_requests import fingerprint, _failed_raw, _safe_failure, _safe_request_metadata


def implementation():
    return {p: digest(APP/p) for p in ('meme_director.py', 'meme_stage.py', 'meme_policy.json', 'meme_verifier.py',
            'gemini_transport.py', 'gemini_stream.py', 'replay_evidence.py', 'llm_transport.py', 'prompts/meme_director.md')}


def focused_indices(evidence, peak):
    """Keep dense contact evidence and sparse context, using actual source PTS."""
    keep={0,len(evidence)-1,min(range(len(evidence)),key=lambda i:abs(evidence[i]['start_sec']-peak))}
    previous=-math.inf
    for i,frame in enumerate(evidence):
        t=frame['start_sec']
        if abs(t-peak)<=.6 or t-previous>=.25:
            keep.add(i);previous=t
    return sorted(keep)


def prepare(manifest_path, directory, *, endpoint='https://aihubmix.com/v1',
            model='qwen3.5-35b-a3b', protocol='chat', fps=12, width=1280,
            max_tokens=8192, reasoning_effort='high'):
    """No credential reads, network requests or model calls in preparation."""
    from .semantic import sampled_evidence
    from .replay_evidence import prepare_visual_context
    manifest_path, directory = Path(manifest_path).resolve(), Path(directory).resolve()
    if directory.exists():
        raise ValueError('Preparation directory already exists; use a new directory')
    if protocol not in ('gemini', 'chat') or not model or not endpoint.startswith('https://'):
        raise ValueError('Invalid explicit model configuration')
    if type(max_tokens) is not int or not 1024 <= max_tokens <= 65536 or reasoning_effort not in ('low','medium','high'):
        raise ValueError('Invalid output budget or thinking level')
    finite_number(fps, 'fps', 8, 16)
    if type(width) is not int or not 960 <= width <= 1600:
        raise ValueError('Frame width must be 960..1600')
    manifest = read_json(manifest_path)
    if manifest.get('version') != 1 or not isinstance(manifest.get('clips'), list) or not 1 <= len(manifest['clips']) <= 20:
        raise ValueError('Expected a version 1 manifest with 1..20 clips')
    video = (manifest_path.parent / manifest['video']).resolve()
    duration = probe(video)['duration_sec']
    raw_assets = read_json((manifest_path.parent / manifest['assets']).resolve())['assets']
    assets_path = (manifest_path.parent / manifest['assets']).resolve()
    assets = {}
    known = {r['id'] for r in policy()['rules']}
    for asset in raw_assets:
        if asset.get('ready') is not True:
            continue
        if asset['id'] not in known or asset['id'] in assets or asset.get('kind') != 'audio':
            raise ValueError('Unknown, duplicate or non-audio ready asset')
        path = (assets_path.parent / asset['path']).resolve()
        begin = finite_number(asset['source_start_sec'], 'asset start', 0, 86400)
        end = finite_number(asset['source_end_sec'], 'asset end', begin+.1, begin+5)
        audio = next((s for s in media_info(path)['streams'] if s['codec_type'] == 'audio'), None)
        if not audio or end > float(audio.get('duration', end)) + .001:
            raise ValueError('Asset phrase exceeds available audio')
        assets[asset['id']] = {**asset, 'path': str(path), 'sha256': digest(path),
                              'source_start_sec': begin, 'source_end_sec': end}
    jobs, identities, clips = [], {}, []
    ids = set()
    for raw in manifest['clips']:
        clip = dict(raw)
        rid = clip.get('id')
        if not isinstance(rid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', rid) or rid in ids:
            raise ValueError('Clip ids must be unique safe identifiers')
        ids.add(rid)
        path = (manifest_path.parent / clip['source']).resolve()
        if str(path) not in identities:
            identities[str(path)] = {**probe(path), 'identity': {'sha256': digest(path)}}
        source = identities[str(path)]
        for field in ('source_start_sec', 'source_end_sec', 'peak_sec', 'context_start_sec', 'context_end_sec'):
            finite_number(clip[field], field, 0, source['duration_sec'])
        if not (clip['context_start_sec'] <= clip['source_start_sec'] < clip['peak_sec'] < clip['source_end_sec'] <= clip['context_end_sec']
                and clip['context_end_sec'] - clip['context_start_sec'] <= 8):
            raise ValueError('Context must contain the entire replay and last at most 8 seconds')
        finite_number(clip['playback_rate'], 'playback_rate', .25, 1)
        finite_number(clip['output_start_sec'], 'output_start_sec', 0, duration)
        finite_number(clip['output_end_sec'], 'output_end_sec', clip['output_start_sec']+.1, duration+.001)
        mapped_length = (clip['source_end_sec'] - clip['source_start_sec'])/clip['playback_rate']
        if abs(clip['output_end_sec'] - clip['output_start_sec'] - mapped_length) > .1:
            raise ValueError('Replay mapping does not match the approved edit')
        constraints = clip.get('editor_constraints', {})
        if not isinstance(constraints, dict) or any(m not in known for m in constraints.get('forbidden_memes', [])):
            raise ValueError('Invalid editor constraints')
        if not isinstance(clip.get('story_context', ''), str):
            raise ValueError('Story context must be an explicit editorial string')
        if 'focus_bbox' in clip:
            box = clip['focus_bbox']
            if (not isinstance(box,list) or len(box)!=4
                    or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in box)
                    or not box[0]<box[2] or not box[1]<box[3]):
                raise ValueError('Focus box must use normalized display coordinates')
        clip['source'] = str(path)
        clips.append(clip)
    ordered = sorted(clips, key=lambda c: c['output_start_sec'])
    if any(a['output_end_sec'] > b['output_start_sec'] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Replay intervals overlap in the output')
    directory.mkdir(parents=True)
    for clip in clips:
        source = identities[clip['source']]
        evidence, content, _ = sampled_evidence(source, clip['context_start_sec'], clip['context_end_sec'],
                                               fps, width, deadline=time.monotonic()+120)
        images=[part for part in content if part['type']=='image_url']
        selected=focused_indices(evidence,clip['peak_sec'])
        evidence=[evidence[i] for i in selected]
        content=[images[i] for i in selected]
        hints = [{'start_sec':clip['context_start_sec'],'end_sec':clip['context_end_sec'],
                  'bbox':clip['focus_bbox'],'coordinate_space':'normalized_display'}] if 'focus_bbox' in clip else None
        # Keep the panorama, while the same native source frame supplies details.
        # Pixel captions bind each visible frame to its id, PTS and target time.
        content, visual_audit = prepare_visual_context(evidence,content,source,
            spatial_hints=hints,overview_frames=0,detail_limit=96,deadline=time.monotonic()+120)
        images = [p for p in content if p['type'] == 'image_url']
        if len(evidence) != len(images) or not 3 <= len(images) <= 128:
            raise ValueError('Invalid evidence count')
        for frame, image in zip(evidence, images):
            path = directory / 'frames' / clip['id'] / (frame['id'] + '.jpg')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(image['image_url']['url'].split(',', 1)[1], validate=True))
            frame.update(image=str(path.relative_to(directory)), sha256=digest(path))
        jobs.append({'clip': clip, 'source_sha256': source['identity']['sha256'],
                     'evidence': evidence, 'visual_context':visual_audit})
    prepared = {'version': 1, 'manifest_sha256': digest(manifest_path), 'video': str(video),
        'video_sha256': digest(video), 'assets': assets, 'jobs': jobs, 'policy': policy(),
        'prompt': (APP/'prompts/meme_director.md').read_text(), 'implementation': implementation(),
        'settings': {'endpoint': endpoint.rstrip('/'), 'model': model, 'protocol': protocol,
                     'fps': fps, 'width': width, 'max_tokens': max_tokens,
                     'reasoning_effort': reasoning_effort, 'timeout_sec': 240},
        'data_scope': {'sampled_frames_only': True, 'audio': False, 'whole_video': False},
        'network_calls': 0, 'model_calls': 0}
    prepared['signature'] = fingerprint(prepared)
    save_json(directory/'prepared.json', prepared)
    return prepared


def checked_preparation(directory):
    prepared = read_json(directory/'prepared.json')
    if prepared['signature'] != fingerprint({k:v for k,v in prepared.items() if k != 'signature'}):
        raise ValueError('Prepared request binding changed')
    if prepared['implementation'] != implementation() or digest(prepared['video']) != prepared['video_sha256']:
        raise ValueError('Code, policy or approved video changed; prepare a new review')
    for asset in prepared['assets'].values():
        if digest(asset['path']) != asset['sha256']:
            raise ValueError('Audio asset changed')
    for job in prepared['jobs']:
        for frame in job['evidence']:
            path = (directory/frame['image']).resolve()
            if not path.is_relative_to(directory) or digest(path) != frame['sha256']:
                raise ValueError('Prepared evidence changed')
    return prepared


def payload(prepared, job, directory):
    clip = job['clip']
    available={k:a for k,a in prepared['assets'].items()
               if k not in clip.get('editor_constraints',{}).get('forbidden_memes',[])}
    active_policy={**prepared['policy'],'rules':[r for r in prepared['policy']['rules'] if r['id'] in available]}
    focus=min(job['evidence'],key=lambda f:abs(f['start_sec']-clip['peak_sec']))
    # Source paths, credentials, past model labels and old meme choices are not sent.
    context = {'policy': active_policy, 'available_memes': [
        {'id': key, 'duration_sec': a['source_end_sec']-a['source_start_sec']} for key,a in available.items()],
        'time_basis': 'source video seconds; original speed, not slow-motion speed',
        'audio_available': False, 'replay_interval': [clip['source_start_sec'], clip['source_end_sec']],
        'focus_event_source_sec': clip['peak_sec'],
        'focus_frame_id': focus['id'],
        'focus_area_normalized_display': clip.get('focus_bbox'),
        'image_layout': 'FULL panorama with frame ID and source PTS; some images append SAME FRAME DETAIL. Both panels are the SAME instant, not two consecutive actions.',
        'playback_rate': clip['playback_rate'], 'editor_constraints': clip.get('editor_constraints', {}),
        'story_context': clip.get('story_context', ''),
        'frames': [{k:f[k] for k in ('id','start_sec')} for f in job['evidence']]}
    content = [{'type':'text', 'text':json.dumps(context, ensure_ascii=False)}]
    # Some compatible gateways downgrade json_schema to json_object. Keep the
    # actual contract in the prompt so the model still sees the required fields.
    content.append({'type':'text', 'text':'Required JSON schema: '+
                    json.dumps(response_schema(active_policy), ensure_ascii=False)})
    for frame in job['evidence']:
        content.extend([{'type':'text', 'text':f'{frame["id"]} source_sec={frame["start_sec"]:.6f}'},
            {'type':'image_url', 'image_url': {'url':'data:image/jpeg;base64,'+
                base64.b64encode((directory/frame['image']).read_bytes()).decode(), 'detail':'high'}}])
    content.extend([{'type':'text','text':f'FOCUS REFERENCE REPEAT: {focus["id"]}, source_sec={focus["start_sec"]:.6f}. This repeats an existing frame, not another instant. Base the target action on the player posture visible here and the adjacent frames. Do not invent airborne feet, an overhead swing, or a spike when the target player is visibly low and grounded.'},
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+
            base64.b64encode((directory/focus['image']).read_bytes()).decode(),'detail':'high'}}])
    return {'model': prepared['settings']['model'], 'max_tokens': prepared['settings']['max_tokens'],
        'reasoning_effort': prepared['settings']['reasoning_effort'],
        'messages': [{'role':'system','content':prepared['prompt']},{'role':'user','content':content}],
        'response_format': {'type':'json_schema','json_schema':{'name':'meme_decision','strict':True,'schema':response_schema(active_policy)}}}


def run_review(directory, key, max_calls, *, request_fn=None):
    """One persistent reservation per call; failures halt, never silently retry."""
    directory = Path(directory).resolve()
    if type(max_calls) is not int or not 1 <= max_calls <= 20 or not isinstance(key, str) or not key:
        raise ValueError('An explicit 1..20 call cap and API key are required')
    with (directory/'review.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Meme review already running') from None
        prepared = checked_preparation(directory)
        settings = prepared['settings']
        native_progress = request_fn is None and settings['protocol'] == 'gemini'
        if request_fn is None:
            if settings['protocol'] == 'gemini':
                from .gemini_transport import request_json as request_fn
            else:
                from .llm_transport import request_json as request_fn
        budget_path = directory/'budget.json'
        budget = {'signature':prepared['signature'], 'max_calls':max_calls, 'retries':0}
        if budget_path.exists() and read_json(budget_path) != budget:
            raise ValueError('Persisted API budget cannot be raised or reset in this run')
        if not budget_path.exists():
            save_json(budget_path, budget)
        ledger_path = directory/'calls.jsonl'
        ledger = [json.loads(s) for s in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
        if len(ledger) > max_calls or any(r['signature'] != prepared['signature'] or r['number'] != i+1 for i,r in enumerate(ledger)):
            raise ValueError('Invalid persisted call ledger')
        if (directory/'halt.json').exists():
            raise ValueError('Previous request failed; this run is halted with no automatic retry')
        results, observations = {}, {}
        def call_once(job, stage, wire, validate):
            rid = job['clip']['id']
            subdirectory = 'results' if stage == 'decision' else 'verifications'
            result_path = directory/subdirectory/f'{rid}.json'
            reservations = [r for r in ledger if r['rally_id'] == rid and r.get('stage','decision') == stage]
            if reservations:
                if len(reservations) != 1 or not result_path.exists():
                    save_json(directory/'halt.json', {'reason':'interrupted_request','automatic_retry':False})
                    raise ValueError('A previously reserved request did not complete')
                cached = read_json(result_path)
                if (cached.get('status') != 'complete' or cached.get('reservation') != reservations[0]
                        or reservations[0]['payload_sha256'] != fingerprint(wire)):
                    raise ValueError('Invalid or failed review cache')
                return validate(cached['decision'])
            if result_path.exists():
                raise ValueError('Unbound review cache without a request reservation')
            if len(ledger) >= max_calls:
                return None
            reservation = {'number':len(ledger)+1, 'signature':prepared['signature'], 'rally_id':rid,
                           'stage':stage,
                           'payload_sha256':fingerprint(wire), 'reserved_unix_sec':time.time()}
            with ledger_path.open('a') as stream:
                stream.write(json.dumps(reservation)+'\n'); stream.flush(); os.fsync(stream.fileno())
            ledger.append(reservation)
            raw = None
            metadata = {}
            try:
                kwargs = {}
                if native_progress:
                    kwargs['progress'] = lambda audit: save_json(directory/'progress'/f'{rid}_{stage}.json',audit)
                raw, metadata = request_fn(settings['endpoint'], key, wire, settings['timeout_sec'], **kwargs)
                safe, redacted = _failed_raw(raw, key)
                if redacted:
                    raise ValueError('meme_sensitive_response')
                decision = validate(safe)
                safe_metadata = _safe_request_metadata(metadata)
                if isinstance(metadata.get('transport'),dict):
                    safe_metadata['transport'] = _failed_raw(metadata['transport'],key)[0]
                save_json(result_path, {'status':'complete', 'reservation':reservation,
                    'decision':decision, 'request':safe_metadata})
                return decision
            except Exception as exc:
                failure = _safe_failure(exc)
                if isinstance(getattr(exc,'diagnostics',None),dict):
                    failure['transport'] = _failed_raw(exc.diagnostics,key)[0]
                if isinstance(exc, ValueError):
                    allowed = {'meme_response_fields','meme_response_enum','meme_response_confidence',
                        'meme_response_reason','meme_unknown_or_unordered_frames','meme_response_facts',
                        'meme_response_fact','meme_contradictory_skip','meme_unknown_id','meme_missing_timing',
                        'meme_sensitive_response','meme_observation_fields','meme_observation_value'}
                    failure['validation_error'] = str(exc) if str(exc) in allowed else 'invalid_meme_decision'
                safe_raw, redacted = _failed_raw(raw if raw is not None else getattr(exc,'final_text',None), key)
                save_json(result_path, {'status':'failed', 'reservation':reservation, 'failure':failure,
                    'raw':safe_raw, 'raw_redacted':redacted, 'request':_safe_request_metadata(metadata), 'reusable':False})
                save_json(directory/'halt.json', {'reason':'request_failed','failure':failure,'automatic_retry':False})
                return None
        from .meme_verifier import observation_payload, validate_observation
        for job in prepared['jobs']:
            rid = job['clip']['id']
            decision = call_once(job, 'decision', payload(prepared, job, directory),
                lambda raw: validate_decision(raw, job['evidence'], prepared['policy']))
            if decision is None:
                break
            results[rid] = decision
            cue, _ = eligible_cue(decision, job['evidence'], job['clip'], prepared['assets'], prepared['policy'])
            if cue and cue['meme_id'] == 'kaipao':
                observation = call_once(job, 'action_verification', observation_payload(prepared, job, directory),
                                        validate_observation)
                if observation is None:
                    break
                observations[rid] = observation
        plan = compile_plan(prepared['jobs'], results, prepared['assets'], prepared['policy'], observations=observations)
        plan.update(prepared_signature=prepared['signature'], policy_sha256=fingerprint(prepared['policy']),
                    source_video_sha256=prepared['video_sha256'], model=settings['model'],
                    api_calls_consumed=len(ledger))
        save_json(directory/'audio_plan.json', plan)
        return plan


def argument_parser():
    parser = argparse.ArgumentParser(description='大模型按梗条件判断配音，证据不足时留白')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='只抽帧和准备请求；不读密钥、不联网')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--api-base', default='https://aihubmix.com/v1')
    p.add_argument('--model', default='qwen3.5-35b-a3b')
    p.add_argument('--protocol', choices=('gemini','chat'), default='chat')
    p.add_argument('--max-output-tokens', type=int, default=8192)
    p.add_argument('--thinking-level', choices=('low','medium','high'), default='high')
    p = sub.add_parser('run', help='按明确次数上限调用 API；错误即停')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--llm-config', type=Path, default=Path('llm_api.json'))
    p.add_argument('--max-calls', type=int, required=True)
    p.add_argument('--render-output', type=Path)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.manifest, args.run, endpoint=args.api_base, model=args.model, protocol=args.protocol,
                         max_tokens=args.max_output_tokens, reasoning_effort=args.thinking_level)
        print(json.dumps({'status':'prepared_offline', 'jobs':len(result['jobs']),
                          'frames':sum(len(j['evidence']) for j in result['jobs']), 'model_calls':0}))
    else:
        prepared = checked_preparation(args.run.resolve())
        config = read_json(args.llm_config).get('llm', {}) if args.llm_config.exists() else {}
        if config.get('base_url', prepared['settings']['endpoint']).rstrip('/') != prepared['settings']['endpoint']:
            raise ValueError('Configured credential endpoint differs from prepared request endpoint')
        key = os.environ.get('VOLLEYMOLE_API_KEY') or config.get('api_key')
        result = run_review(args.run, key, args.max_calls)
        if args.render_output:
            if not result['review_complete']:
                raise RuntimeError('Meme review incomplete; original edit retained, no fallback memes rendered')
            from .meme_audio import render
            render(prepared['video'], args.run/'audio_plan.json', args.render_output)
        print(json.dumps({'status':'complete' if result['review_complete'] else 'incomplete',
                          'cues':len(result['cues']), 'calls_consumed':result['api_calls_consumed']}))


if __name__ == '__main__':
    main()
