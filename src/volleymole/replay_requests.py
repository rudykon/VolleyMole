"""Reusable, strict visual requests independent of replay selection policy.

A changed expansion budget should reuse an identical visual question. A changed
source, question, prompt, sampler, image layout or model must ask it again.
Credentials and local retry/deadline policy are never part of cache identity.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError

from .common import APP, digest, read_json, save_json
from .llm_transport import ResponseContractError, TransportError, is_retryable_transport_error, _safe_usage
from .replay_evidence import prepare_visual_context
from .replay_review_schema import payload
VERSION = 1

_VALIDATION_CODES = frozenset((
    'replay_action', 'replay_boundary_boolean', 'replay_candidate_count',
    'replay_candidate_order', 'replay_confidence', 'replay_contradictory_accept',
    'replay_excitement', 'replay_expansion_direction', 'replay_frame_evidence',
    'replay_response_fields', 'replay_response_text', 'replay_review_order',
    'replay_unknown_frame', 'replay_verdict', 'replay_verify_event_order',
    'replay_verify_event_type', 'replay_verify_missing_event_frames',
    'replay_verify_motion_boolean', 'replay_verify_next_touch_order'))
_FINISH_REASONS = ('stop', 'length', 'content_filter', 'tool_calls', 'other', None)


def _validation_code(error):
    # Never reflect a provider's arbitrary ValueError text into a new prompt.
    if type(error) is ValueError and str(error) in _VALIDATION_CODES:
        return str(error)
    return 'replay_invalid_contract' if isinstance(error, ValueError) else None


def _safe_request_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    safe = {}
    enums = dict(finish_reason=_FINISH_REASONS,
                 response_format=('json_schema','json_object','text',None),
                 structured_output_degraded=('model_unsupported','json_object_unsupported_on_anthropic',
                    'json_schema_missing_schema','schema_keywords_stripped','other',None),
                 reasoning_effort=('none','minimal','low','medium','high','xhigh','max',None),
                 response_content_encoding=('json','fenced_json'))
    for key, values in enums.items():
        if key in metadata and metadata[key] in values:
            safe[key] = metadata[key]
    if type(metadata.get('gateway_json_repaired')) is bool:
        safe['gateway_json_repaired'] = metadata['gateway_json_repaired']
    schema_hash = metadata.get('response_schema_sha256')
    if isinstance(schema_hash,str) and re.fullmatch(r'[0-9a-f]{64}',schema_hash):
        safe['response_schema_sha256'] = schema_hash
    usage = _safe_usage(metadata.get('usage'))
    if usage is not None:
        safe['usage'] = usage
    return safe


def _safe_failure(error):
    diagnostic = dict(error=type(error).__name__)
    if isinstance(error, HTTPError) and type(error.code) is int and 100 <= error.code <= 599:
        diagnostic['http_status'] = error.code
    if isinstance(error, TransportError):
        allowed = ('http_error','timeout','connection_error','io_error','authentication_failed','rate_limited','server_error')
        diagnostic['transport_error'] = error.reason if error.reason in allowed else 'other'
        if type(error.http_status) is int and 100 <= error.http_status <= 599:
            diagnostic['http_status'] = error.http_status
    if isinstance(error, ResponseContractError):
        allowed = ('invalid_json','invalid_response_envelope','incomplete_response','refused','response_too_large')
        diagnostic['response_contract'] = error.reason if error.reason in allowed else 'other'
        diagnostic['finish_reason'] = error.finish_reason if error.finish_reason in _FINISH_REASONS else 'other'
    code = _validation_code(error)
    if code:
        diagnostic['validation_error'] = code
    return diagnostic


def _failed_raw(raw, key):
    """Copy model JSON, redacting any echoed credentials or image/request data."""
    redacted = False
    private_names = {'api_key','authorization','access_token','cookie','headers','wire',
                     'messages','image_url','request_body','payload'}
    def clean(value, depth=0):
        nonlocal redacted
        if depth > 30:
            redacted = True
            return '[redacted excessive nesting]'
        if isinstance(value, dict):
            result = {}
            for name, child in value.items():
                if not isinstance(name,str):
                    redacted = True
                    continue
                safe_name = clean(name,depth+1)
                if name.lower() in private_names:
                    result[safe_name] = '[redacted sensitive field]'; redacted = True
                else:
                    result[safe_name] = clean(child,depth+1)
            return result
        if isinstance(value,list):
            return [clean(child,depth+1) for child in value]
        if isinstance(value,str):
            if isinstance(key,str) and key and key in value:
                value = value.replace(key,'[redacted credential]'); redacted = True
            if re.search(r'data:[^\s,]+;base64,|[A-Za-z0-9+/]{256,}={0,2}',value):
                redacted = True
                return '[redacted encoded content]'
            return value
        if value is None or type(value) in (bool,int) or type(value) is float and math.isfinite(value):
            return value
        redacted = True
        return '[redacted non-JSON value]'
    return clean(raw), redacted


def _failed_attempt(cache, signature, source, job, attempt, raw, received, metadata, diagnostic, key):
    safe_raw, redacted = _failed_raw(raw,key)
    path = Path(cache)/'failed_requests'/signature/f'{time.time_ns()}_attempt_{attempt:02d}.json'
    save_json(path,dict(schema_version=1,status='failed',reusable=False,signature=signature,
        source_identity_sha256=fingerprint(source['identity']),job_sha256=fingerprint(job),
        attempt=attempt,validation_error=diagnostic.get('validation_error'),failure=diagnostic,
        raw=safe_raw,raw_available=received,raw_redacted=redacted,request=_safe_request_metadata(metadata)))
    return str(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def request_implementation(phase):
    if phase not in ('scan', 'review', 'verify'):
        raise ValueError('replay_request_phase')
    files = ('replay_requests.py', 'replay_evidence.py', 'replay_review_schema.py',
             'semantic.py', 'llm_transport.py', f'prompts/replay_{phase}.md')
    return {name: digest(APP/name) for name in files}


def evidence_valid(evidence, start, end, fps, source):
    """Require actual ordered frames to cover the entire requested interval."""
    from fractions import Fraction
    try:
        native = 1 / float(Fraction(source.get('nominal_fps', '30/1')))
        frames = [r for r in evidence if r['kind'] == 'frame']
        times = [r['start_sec'] for r in frames]
        ids = [r['id'] for r in frames]
        tolerance = max(1.5/fps, 2*native, .12)
        valid = (native > 0 and math.isfinite(native) and len(times) >= 3
                 and len(frames) == len(evidence) and len(ids) == len(set(ids))
                 and all(type(t) in (int, float) and math.isfinite(t) and start-1e-6 <= t < end for t in times)
                 and times[0]-start <= tolerance+1e-6 and end-times[-1] <= tolerance+1e-6
                 and all(0 < b-a <= tolerance+1e-6 for a, b in zip(times, times[1:])))
    except (TypeError, KeyError, ValueError, ZeroDivisionError, OverflowError):
        valid = False
    if not valid:
        raise ValueError('replay_sampling_gap')


def request_settings(settings, phase):
    """Only values which alter an actual provider request or supplied images."""
    fps = settings['scan_fps'] if phase == 'scan' else settings.get('verify_fps', settings['review_fps']) if phase == 'verify' else settings['review_fps']
    return dict(endpoint=settings['endpoint'], model=settings['model'], max_tokens=settings['max_tokens'],
                reasoning_effort=settings.get('reasoning_effort'), fps=fps, width=settings['width'],
                overview_frames=settings.get('overview_frames', 12), detail_limit=settings.get('detail_limit', 12))


def request_phase(source, job, settings, cache, deadline, *, decoder, request_fn, sample_fn):
    """Return validated result and request audit, using injected I/O/decoder.

    ``job['phase']`` selects ``prompts/replay_<phase>.md``. ``spatial_hints`` uses
    the normalized-display schema of ``prepare_visual_context``. The complete
    job is cache-bound; policy controls outside the job cannot invalidate it.
    ``decoder(raw, evidence)`` must validate the unmodified provider response.
    """
    phase = job['phase']; public = request_settings(settings, phase); fps = public['fps']
    signature_data = dict(version=VERSION, source=source['identity'],
        source_clock={k: source.get(k) for k in ('start_sec', 'rotation', 'nominal_fps', 'duration_sec')},
        job=job, settings=public, implementation=request_implementation(phase))
    signature = fingerprint(signature_data)
    path = Path(cache)/'requests'/f'{signature}.json'
    if path.is_file():
        try:
            saved = read_json(path)
            if saved['signature'] == signature and saved['identity'] == signature_data:
                evidence_valid(saved['evidence'], job['start'], job['end'], fps, source)
                # A complete audit must describe the same source frames. The
                # higher-level rally cache additionally binds this file's hash.
                if (saved['visual_context']['frame_count'] != len(saved['evidence'])
                        or not isinstance(saved['image_sha256'], list) or not saved['image_sha256']):
                    raise ValueError('replay_request_cache_audit')
                result = decoder(saved['raw'], saved['evidence'])
                return result, {**saved, 'cached': True, 'artifact': str(path)}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if time.monotonic() >= deadline:
        raise TimeoutError('replay_review_deadline')
    evidence, content, _ = sample_fn(source, job['start'], job['end'], fps, public['width'], deadline=deadline)
    evidence_valid(evidence, job['start'], job['end'], fps, source)
    content, visual_audit = prepare_visual_context(evidence, content, source,
        spatial_hints=job.get('spatial_hints'), overview_frames=public['overview_frames'],
        detail_limit=public['detail_limit'], deadline=deadline)
    prompt = (APP/'prompts'/f'replay_{phase}.md').read_text(encoding='utf-8')
    image_hashes = [fingerprint(c['image_url']) for c in content if c['type'] == 'image_url']
    # Keep previously used fields stable while allowing verifier-specific data.
    # Spatial hints are already represented in images and the separate audit.
    request_context = {k:v for k,v in job.items() if k not in ('start', 'end', 'candidate', 'spatial_hints')}
    request_context.update(phase=phase, context=[job['start'], job['end']], requested_fps=fps,
        candidate=({k:v for k,v in job['candidate'].items() if not k.endswith('_frame_id')}
                   if job.get('candidate') else None),
        previous_boundary_problem=job.get('boundary_problem'), audio_available=False,
        time_basis='source-local seconds; actual frame PTS')
    content.insert(0, dict(type='text', text=json.dumps(request_context, ensure_ascii=False, allow_nan=False)))
    wire = payload(public['model'], phase, prompt, content, evidence,
                   public['max_tokens'], public['reasoning_effort'])
    failures = []; began = time.monotonic()
    for attempt in range(settings['retries']+1):
        remaining = min(settings['timeout'], deadline-time.monotonic())
        if remaining <= 0:
            raise TimeoutError('replay_review_deadline')
        raw = None; metadata = None; received = False
        try:
            raw, metadata = request_fn(settings['endpoint'], settings['key'], wire, remaining)
            received = True
            result = decoder(raw, evidence)
            break
        except Exception as exc:
            diagnostic = _safe_failure(exc)
            try:
                diagnostic['artifact'] = _failed_attempt(cache,signature,source,job,attempt+1,
                    raw,received,metadata,diagnostic,settings['key'])
            except OSError:
                # A diagnostic write failure must not hide the original API or
                # protocol failure, nor turn it into a completed review.
                diagnostic['artifact_write_error'] = 'OSError'
            failures.append(diagnostic)
            if isinstance(exc, HTTPError):
                exc.close()
            contract = isinstance(exc, (ValueError, ResponseContractError))
            if attempt == settings['retries'] or not (contract or is_retryable_transport_error(exc)):
                raise
            if contract:
                wire['messages'] = wire['messages'][:2] + [dict(role='user', content=
                    '上一响应未通过本地协议校验（'+diagnostic['validation_error']+'）。请重新查看原始画面，严格使用提供的帧号，满足时间顺序与当前阶段的 JSON Schema；不要修补上一响应或把不确定的动作判为已通过。')]
    saved = dict(signature=signature, identity=signature_data, raw=raw, evidence=evidence,
        image_sha256=image_hashes, visual_context=visual_audit, request=metadata,
        retry_history=failures, elapsed_sec=round(time.monotonic()-began, 3))
    save_json(path, saved)
    return result, {**saved, 'cached': False, 'artifact': str(path)}
