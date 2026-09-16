"""Visual evidence for text-only ranking models, with resumable small batches."""
import base64
import hashlib
import json
import math
import time
import io
import wave
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.request import Request, urlopen
from .common import APP, digest, read_json, save_json
from .schemas import local_file
from .llm_transport import (ResponseContractError, TransportError,
                            is_retryable_transport_error, request_json as transport_request_json)

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


class EventContractError(ResponseContractError):
    """Rejected event data, with safe wire-contract diagnostics but no body."""
    def __init__(self, validation_error, metadata):
        super().__init__('invalid_event_contract', metadata.get('finish_reason'))
        self.validation_error = validation_error
        self.request_metadata = {k: metadata[k] for k in
            ('response_format', 'structured_output_degraded', 'response_schema_sha256', 'reasoning_effort',
             'gateway_json_repaired', 'response_content_encoding')
            if k in metadata}


def request_json(endpoint, key, payload, timeout):
    # Preserve legacy HTTP classification in ranker/api_probe; event failures
    # redact it at the bounded-map boundary. Keep urlopen injectable for tests.
    return transport_request_json(endpoint, key, payload, timeout, opener=urlopen,
                                  preserve_http_error=True)


def failure_diagnostics(exc):
    """Persist only allowlisted diagnostics, never arbitrary exception text."""
    from urllib.error import HTTPError
    failure = {'error': type(exc).__name__}
    if isinstance(exc, HTTPError): failure['http_status'] = exc.code
    if isinstance(exc, TransportError):
        failure['transport_error'] = exc.reason
        if exc.http_status is not None: failure['http_status'] = exc.http_status
    if isinstance(exc, ResponseContractError):
        failure.update(response_contract=exc.reason, finish_reason=exc.finish_reason)
    if isinstance(exc, EventContractError):
        failure.update(validation_error=exc.validation_error, request=exc.request_metadata)
    return failure


def candidate_fields(rally, full_evidence=False):
    fields = {**{k:rally[k] for k in ('rally_id', 'start_sec', 'end_sec', 'safe_start_sec', 'safe_end_sec',
                                'duration_sec', 'actions', 'players', 'ball_metrics', 'rule_score')},
            'uncertainty':rally.get('uncertainty',{'note':'No extra fusion audit in this historical candidate.'})}
    if 'source_id' in rally:
        fields.update(source_id=rally['source_id'],source_set=rally['source_set'],time_basis='seconds within this set, not whole-match time')
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


def sampled_evidence(source, start, end, fps, width=512, audio=None, deadline=None, frame_cache=None):
    """Explicit PTS-labelled video sequence, avoiding provider-default video FPS.

    Decode once per bounded context; the same frames serve motion measurements
    and both semantic collections. Audio is cut from the shared 32 kHz waveform.
    """
    import av
    import cv2
    import numpy as np
    evidence, content, frames = [], [], []
    def decode_context():
        origin = source.get('start_sec', 0.)
        with av.open(source['path']) as container:
            stream = container.streams.video[0]
            container.seek(max(0, int((start+origin)*av.time_base)), backward=True)
            next_time = start
            for frame in container.decode(stream):
                if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('evidence deadline')
                if frame.pts is None: continue
                when = float(frame.pts*frame.time_base)-origin
                if when >= end: break
                if when+1e-6 < next_time: continue
                pixels = frame.to_ndarray(format='bgr24')
                rotation = source.get('rotation', 0)
                if rotation == 90: pixels = cv2.rotate(pixels, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180: pixels = cv2.rotate(pixels, cv2.ROTATE_180)
                elif rotation == 270: pixels = cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE)
                pixels = cv2.resize(pixels, (width, max(2, round(pixels.shape[0]*width/pixels.shape[1]))))
                yield when, pixels
                next_time = start+(math.floor((when-start)*fps+1e-6)+1)/fps
    sampled = None
    if frame_cache:
        from .event_frames import frames as shared_frames
        sampled = shared_frames(frame_cache, start, end, fps, width, deadline)
    if sampled is None: sampled = decode_context()
    for when, pixels in sampled:
        ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok: raise ValueError('证据编码失败')
        ref = f'frame_{len(evidence):05d}'
        evidence.append({'id': ref, 'kind': 'frame', 'start_sec': when, 'end_sec': when})
        content.extend([{'type': 'text', 'text': f'{ref} source_sec={when:.6f}'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode(), 'detail': 'low' if width <= 512 else 'high'}}])
        frames.append((when, cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)))
    if not frames: raise ValueError('上下文没有视频帧')
    # Reject truncated spools/decodes instead of calling a partial view complete.
    nominal = source.get('nominal_fps', '30/1')
    try:
        from fractions import Fraction
        source_interval = 1/float(Fraction(nominal))
    except (ValueError, ZeroDivisionError): source_interval = .1
    tolerance = max(2/fps, 2*source_interval, .25)
    if frames[0][0]-start > tolerance or end-frames[-1][0] > tolerance:
        raise ValueError('实际采样未覆盖请求上下文首尾')
    if audio is not None:
        samples = audio[round(start*32000):round(end*32000)]
        if len(samples):
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(32000)
                out.writeframes((np.clip(samples, -1, 1)*32767).astype('<i2').tobytes())
            evidence.append({'id': 'audio', 'kind': 'audio', 'start_sec': start, 'end_sec': start+len(samples)/32000})
            content.extend([{'type': 'text', 'text': f'audio begins at source_sec={start}; synchronized WAV'},
                {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(buffer.getvalue()).decode(), 'format': 'wav'}}])
    return evidence, content, frames


def bounded_map(jobs, function, concurrency, deadline, stopped=None):
    """Bound queue depth as well as running work using a shared phase deadline.

    Workers receive the same absolute deadline. Never abandon workers writing
    caches behind the caller; each completes or times out before pool shutdown.
    """
    pending, results, failures = {}, [], []
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < concurrency and time.monotonic() < deadline and not (stopped and stopped()):
                job = next(iterator, None)
                if job is None: exhausted = True; break
                pending[pool.submit(function, job)] = job
            if not pending: break
            done, _ = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try: results.append(future.result())
                except Exception as exc:
                    # Do not persist provider bodies or secrets in error messages.
                    from urllib.error import HTTPError
                    failure = {'job': job['id'], 'status': 'unreviewed', **failure_diagnostics(exc)}
                    if isinstance(exc, HTTPError): exc.close()
                    if hasattr(exc, 'attempts'): failure['attempts'] = exc.attempts
                    if hasattr(exc, 'retry_history'): failure['retry_history'] = exc.retry_history
                    failures.append(failure)
            if time.monotonic() >= deadline or (stopped and stopped()): exhausted = True
    return results, failures


def understand_context(job, source, cache, settings, audio, audio_features, deadline, records=None):
    import fcntl
    lock_key = hashlib.sha256(json.dumps({'source': source['identity'], 'job': job,
        'settings': {k:v for k,v in settings.items() if k not in ('key','frame_cache')}}, sort_keys=True).encode()).hexdigest()
    lock_path = Path(cache)/'locks'/f'{lock_key}.lock'; lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        while True:
            try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() >= deadline: raise TimeoutError('semantic cache lock deadline')
                time.sleep(.1)
        return _understand_context(job, source, cache, settings, audio, audio_features, deadline, records)


def _understand_context(job, source, cache, settings, audio, audio_features, deadline, records=None):
    from .events import validate_events
    from .motion_features import camera_transform, relative_motion, ball_motion
    import bisect
    import numpy as np
    prompt = (APP/'prompts/understand_events.md').read_text(encoding='utf-8')
    signature_data = {'version': 1, 'source': source['identity'], 'job': job,
        'settings': {k: v for k, v in settings.items() if k not in ('key','frame_cache')}, 'prompt': prompt,
        'implementation': [digest(APP/name) for name in ('semantic.py', 'llm_transport.py', 'events.py', 'event_schema.py', 'audio_events.py', 'motion_features.py', 'event_frames.py','action_evidence.py')],
        'local_signature': settings.get('local_signature') if records is not None else None}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    path = Path(cache)/f'{signature}.json'
    if path.is_file():
        try:
            saved = read_json(path)
            if saved['signature'] == signature:
                validate_events({'events': saved['events']}, saved['evidence'], job['start'], job['end'])
                return {**saved, 'cached': True}
        except (ValueError, KeyError, TypeError): pass
    fps = settings['coarse_fps'] if job['phase'] == 'coarse' else settings['review_fps']
    width = 512 if job['phase'] == 'coarse' else 768
    if settings.get('frame_width') is not None:
        width = min(width, settings['frame_width'])
    evidence, content, frames = sampled_evidence(source, job['start'], job['end'], fps,
        width,
        audio if settings['modality'] == 'frames-audio' else None, deadline, settings.get('frame_cache'))
    local = {'audio_status': audio_features['status'], 'sound_events': [], 'audio_windows': [], 'motion': []}
    from .action_evidence import context_hypotheses,evidence_status
    local['action_hypotheses']=context_hypotheses(audio_features.get('action_model',{}),job['start'],job['end'])
    local['action_model_status']=audio_features.get('action_model',{}).get('status','not_configured')
    local['action_model_coverage']=evidence_status(audio_features.get('action_model',{}))
    for i, event in enumerate(audio_features.get('sound_events', [])):
        if job['start'] <= event['start_sec'] < event['end_sec'] <= job['end']:
            ref = f'sound_{i}'
            evidence.append({'id': ref, 'kind': 'sound_event', 'start_sec': event['start_sec'], 'end_sec': event['end_sec']})
            local['sound_events'].append({**event, 'id': ref})
    local['audio_windows'] = [w for w in audio_features['windows'] if job['start'] <= w['start_sec'] < job['end']]
    if records:
        times = [r['source_time_s']-source.get('start_sec', 0) if 'source_time_s' in r else r['time_s'] for r in records]
        previous = None
        for when, gray in frames:
            pos = min(bisect.bisect_left(times, when), len(times)-1)
            people = []
            for p in records[pos].get('players', []):
                scaled = {**p, 'xyxy': (np.array(p['xyxy'])*gray.shape[1]/source['width']).tolist()}
                if p.get('keypoints'):
                    kp = np.array(p['keypoints']); kp[:, :2] *= gray.shape[1]/source['width']; scaled['keypoints'] = kp.tolist()
                people.append(scaled)
            ball = records[pos].get('event_ball')
            if ball is not None: ball = (np.asarray(ball)*gray.shape[1]/source['width']).tolist()
            if previous:
                before, old_gray, old_people, old_ball = previous
                camera = camera_transform(old_gray, gray, [p['xyxy'] for p in old_people])
                measurement = relative_motion(old_people, people, when-before, camera)
                movement = ball_motion(old_ball, ball, when-before, camera, people)
                transient = any(w['transient_candidate'] and w['start_sec']-.15 <= when <= w['end_sec']+.15 for w in local['audio_windows'])
                measurement['ball'] = movement
                measurement['touch_hypothesis'] = True if movement['near_wrist'] is True and transient else None
                ref = f'motion_{len(local["motion"])}'
                evidence.append({'id': ref, 'kind': 'local_motion', 'start_sec': before, 'end_sec': when,
                    'measured': measurement['body_lengths_per_sec'] is not None})
                local['motion'].append({'id': ref, 'time_sec': when, **measurement})
            previous = (when, gray, people, ball)
    content.insert(0, {'type': 'text', 'text': json.dumps({'phase': job['phase'], 'context': [job['start'], job['end']],
        'requested_fps': fps, 'actual_sample_times': [t for t, _ in frames], 'local': local,
        'candidate': job.get('candidate'), 'evidence': evidence}, ensure_ascii=False)})
    from .event_schema import event_request_payload, decode_event_response, WIRE_PROTOCOL
    payload = event_request_payload(settings['model'], prompt, content, evidence, job['start'], job['end'],
        settings.get('max_tokens', 4096), settings.get('reasoning_effort'))
    began = time.monotonic()
    original_messages = payload['messages']
    retry_history = []
    for attempt in range(settings['retries']+1):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            exc = TimeoutError('analysis deadline')
            exc.attempts, exc.retry_history = attempt, retry_history
            raise exc
        data = None
        try:
            data, metadata = request_json(settings['endpoint'], settings['key'], payload, min(settings['timeout'], remaining))
            try:
                events = decode_event_response(data, evidence, job['start'], job['end'])['events']
            except (ValueError, TypeError, KeyError) as exc:
                # Anchor compilation never fills in missing facts/scores or
                # accepts invalid data, even after a gateway schema downgrade.
                reason = str(exc) if type(exc) is ValueError else 'invalid_event_shape'
                raise EventContractError(reason, metadata) from None
            break
        except Exception as exc:
            from urllib.error import HTTPError
            retryable = is_retryable_transport_error(exc)
            contract_retry = isinstance(exc, EventContractError) or (
                isinstance(exc, ResponseContractError) and (exc.reason in ('invalid_json', 'invalid_response_envelope')
                or (exc.reason == 'incomplete_response' and exc.finish_reason == 'length')))
            retry_history.append({'attempt': attempt+1, **failure_diagnostics(exc)})
            exc.attempts, exc.retry_history = attempt+1, list(retry_history)
            if isinstance(exc, HTTPError): exc.close()
            if not (retryable or contract_retry) or attempt == settings['retries']: raise
            if contract_retry:
                messages = list(original_messages)
                if data is not None:
                    prior = json.dumps(data, ensure_ascii=False)
                    if len(prior) <= 32000: messages.append({'role': 'assistant', 'content': prior})
                detail = exc.validation_error if isinstance(exc, EventContractError) else exc.reason
                messages.append({'role': 'user', 'content':
                    '上一响应未通过本地校验：'+detail+'。请根据同一批原始证据重新输出完整的 '+WIRE_PROTOCOL+' JSON。'
                    '事实 frame_id 选已有画面，support_id 仅选该帧附近的辅助证据或 null；'
                    '评分 fact_indexes 是本事件 facts 数组中从 0 开始的索引。'
                    'dimensions 每项必须有 dimension、value、fact_indexes 三个字段，九个维度名各出现一次，不能只给无名分数数组。'
                    '不得为通过校验编造事实、修改实际观察或把无效事件机械改为空数组。'})
                payload = {**payload, 'messages': messages}
                if exc.finish_reason == 'length':
                    payload['max_tokens'] = min(16384, payload['max_tokens']*2)
            # Finite jitter-free backoff, bounded by the absolute deadline.
            delay = min(2**attempt, max(0, deadline-time.monotonic()))
            time.sleep(delay)
    result = {'signature': signature, 'job': job, 'events': events, 'evidence': evidence,
        'local': local, 'request': metadata, 'attempts': attempt+1, 'elapsed_sec': time.monotonic()-began,
        'wire_protocol': WIRE_PROTOCOL, 'wire_response': data, 'retry_history': retry_history,
        'dimension_encodings': ['named_entries' if isinstance(e['dimensions'], list) else 'named_object'
                                for e in data['events']],
        'time_derivation': 'exact frame PTS; clip envelope includes explicit anchors and cited fact frames',
        'requested_fps': fps, 'sample_times_sec': [t for t, _ in frames], 'cached': False}
    save_json(path, result)
    return result
