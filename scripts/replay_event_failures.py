#!/usr/bin/env python3
"""Replay chosen historical event jobs through the production GLM frames path.

Uploads sampled pictures to the configured provider, never audio. Every run
requires a new output directory; old semantic responses cannot satisfy a replay.
Valid or nonempty output is a protocol result, not recognition ground truth.
"""
import argparse
import copy
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volleymole.common import APP, digest, read_json, save_json
from volleymole.events import windows
from volleymole.semantic import (understand_context,
                                failure_diagnostics as production_failure_diagnostics)

MODEL = 'glm-5.3-flash'
VALIDATION_ERRORS = {
    'invalid_event_shape', '事件响应必须包含 events 数组', '单块事件过多',
    '事件字段不完整', '事件时间越过已观察上下文', '事件剪辑边界裁断过程',
    '回放峰值不属于事件', '无效事件置信度', '状态必须为布尔或未知',
    '无效事件说明', '事实必须为数组', '事实必须带时间和原始证据',
    '无效事实描述', '事实不在事件上下文内', '事件引用了不存在的证据',
    '事实时间与证据不对应', '事件没有直接观察', '事件缺少画面证据',
    '双榜维度不完整', '维度必须带证据', '量表范围为 0–4 或未知',
    '评分必须基于已提取事实', '笑声评分缺少音画关联',
    '运动强度缺少可靠相机补偿测量',
    '事件理解缺少画面证据', '事件理解缺少输入证据', '单块锚点事件过多',
    '输入证据 ID 重复', '帧锚必须引用本上下文的画面证据', '锚点事件字段不完整',
    '事件帧锚顺序错误', '事件事实数量错误', '锚点事实字段不完整', '无效事实类别',
    '事实辅助证据必须为本上下文的声音或运动测量', '事实帧时间与辅助证据不对应',
    '维度必须引用事实索引', '评分引用了不存在的事实索引',
    '未知评分不能附带事实索引', '非未知评分必须引用事实且取值为 0–4',
    '维度必须显式命名且引用事实索引', '双榜维度名称重复或无效',
}
ERROR_TYPES = {'EventContractError', 'ResponseContractError', 'TransportError',
    'HTTPError', 'URLError', 'TimeoutError', 'RemoteDisconnected', 'ConnectionError',
    'ConnectionResetError', 'BrokenPipeError', 'IncompleteRead', 'OSError',
    'FileNotFoundError', 'PermissionError', 'ValueError', 'TypeError', 'KeyError',
    'IndexError', 'RuntimeError', 'JSONDecodeError'}


def restore_jobs(timeline, manifest, names, max_review_sec=90):
    """Prefer recorded requests; otherwise reconstruct the original 24/4 jobs."""
    duration = timeline['source']['duration_sec']
    recorded = {r['job']['id']: r['job'] for r in timeline.get('requests', [])}
    rallies = {r['rally_id']: r for r in manifest['rallies']}
    chunks = list(windows(duration, 24, 4))
    if len(set(names)) != len(names):
        raise ValueError('Duplicate job IDs are not allowed')
    jobs = []
    for name in names:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
            raise ValueError('Job IDs must be safe file names')
        if name in recorded:
            job = copy.deepcopy(recorded[name])
        elif re.fullmatch(r'coarse_\d{5}', name):
            index = int(name.split('_')[1])
            if index >= len(chunks):
                raise ValueError('Coarse job is outside the source')
            start, end = chunks[index]
            job = {'id': name, 'start': start, 'end': end, 'phase': 'coarse'}
        elif name in rallies:
            rally = rallies[name]
            job = {'id': name, 'start': max(0, rally['safe_start_sec']-2),
                   'end': min(duration, rally['safe_end_sec']+3), 'phase': 'review',
                   'candidate': {k: rally.get(k) for k in
                                 ('start_sec', 'end_sec', 'actions', 'action_events')}}
        else:
            raise ValueError('Job has neither a saved request nor a local rally')
        if (job.get('phase') not in ('coarse', 'review') or
                any(type(job.get(k)) not in (int, float) or not math.isfinite(job[k])
                    for k in ('start', 'end')) or
                not 0 <= job['start'] < job['end'] <= duration):
            raise ValueError('Job context must lie within the source')
        if job['end']-job['start'] > (24 if job['phase'] == 'coarse' else max_review_sec):
            raise ValueError('Job context exceeds the permitted phase duration')
        jobs.append(job)
    return jobs


def load_records(run, source):
    """Use exactly the local-record/visible-ball association used in finish()."""
    analytics = run/'analytics/detections.jsonl'
    if not analytics.is_file():
        raise ValueError('Review replay requires the original analytics records')
    with analytics.open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream]
    ball_path = run/'tracking/ball.csv'
    if ball_path.is_file():
        with ball_path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                index = int(row['Frame'])
                xy = [float(row['X']), float(row['Y'])]
                if (0 <= index < len(records) and int(row['Visibility']) == 1
                        and all(math.isfinite(x) for x in xy)
                        and 0 <= xy[0] < source['width'] and 0 <= xy[1] < source['height']):
                    records[index]['event_ball'] = xy
    return records, [digest(analytics), digest(ball_path) if ball_path.is_file() else None]


def safe_request(metadata):
    """Do not copy arbitrary provider metadata into diagnostic summaries."""
    result = {}
    enums = {
        'finish_reason': ('stop', 'length', 'content_filter', 'tool_calls', None),
        'response_format': ('json_schema', 'json_object', None),
        'reasoning_effort': ('low', 'high', 'max', None),
        'structured_output_degraded': (None, 'model_unsupported',
            'json_object_unsupported_on_anthropic', 'json_schema_missing_schema',
            'schema_keywords_stripped', 'other'),
        'response_content_encoding': ('json', 'fenced_json'),
        'content_encoding': ('json', 'fenced_json'),
    }
    for key, choices in enums.items():
        if key in metadata:
            result[key] = metadata[key] if metadata[key] in choices else 'other'
    signature = metadata.get('response_schema_sha256')
    if isinstance(signature, str) and re.fullmatch(r'[0-9a-f]{64}', signature):
        result['response_schema_sha256'] = signature
    if type(metadata.get('gateway_json_repaired')) is bool:
        result['gateway_json_repaired'] = metadata['gateway_json_repaired']
    return result


def safe_failure_fields(raw):
    """Recheck the production diagnostic map before saving any replay output."""
    error = raw.get('error')
    result = {'error': error if isinstance(error, str) and error in ERROR_TYPES else 'OtherError'}
    status = raw.get('http_status')
    if type(status) is int and 100 <= status <= 599:
        result['http_status'] = status
    if 'transport_error' in raw:
        reason = raw['transport_error']
        result['transport_error'] = reason if reason in (
            'http_error', 'timeout', 'connection_error', 'io_error') else 'other'
    if 'response_contract' in raw:
        reason = raw['response_contract']
        result['response_contract'] = reason if reason in (
            'invalid_event_contract', 'invalid_json', 'refused', 'incomplete_response',
            'response_too_large', 'invalid_response_envelope') else 'other'
    if 'finish_reason' in raw:
        result['finish_reason'] = safe_request(raw)['finish_reason']
    if 'validation_error' in raw:
        reason = raw['validation_error']
        result['validation_error'] = (reason if isinstance(reason, str) and reason in
                                      VALIDATION_ERRORS else 'other_event_contract_error')
    if isinstance(raw.get('request'), dict):
        result['request'] = safe_request(raw['request'])
    return result


def safe_retry_history(history):
    if not isinstance(history, list):
        return []
    result = []
    for item in history[:100]:
        if not isinstance(item, dict):
            continue
        safe = safe_failure_fields(item)
        if type(item.get('attempt')) is int and 0 <= item['attempt'] <= 100:
            safe['attempt'] = item['attempt']
        result.append(safe)
    return result


def failure_diagnostics(exc):
    result = safe_failure_fields(production_failure_diagnostics(exc))
    if isinstance(exc, HTTPError):
        exc.close()  # Never read provider bodies: they can echo credentials/input.
    attempts = getattr(exc, 'attempts', None)
    result['attempts'] = attempts if type(attempts) is int and 0 <= attempts <= 100 else None
    if hasattr(exc, 'retry_history'):
        result['retry_history'] = safe_retry_history(exc.retry_history)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing single-source part directory')
    parser.add_argument('--jobs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True, help='Must not exist, even if empty')
    parser.add_argument('--model', choices=(MODEL,), default=MODEL)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--reasoning-effort', choices=('low', 'high', 'max'), default='low')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--max-review-sec', type=float, default=90)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('Use a new output directory; previous measurements must be retained')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Timeout must be finite and positive')
    if not 0 <= args.retries <= 5 or not 1 <= args.concurrency <= 6:
        parser.error('Retries must be 0–5 and concurrency 1–6')
    if not 64 <= args.width <= 4096 or not 256 <= args.max_tokens <= 16384:
        parser.error('Width must be 64–4096 and max-tokens 256–16384')
    if not math.isfinite(args.max_review_sec) or not 0 < args.max_review_sec <= 90:
        parser.error('Max-review-sec must be finite and in (0, 90]')
    run = args.run.resolve()
    try:
        timeline = read_json(run/'event_timeline.json')
        manifest = read_json(run/'match_manifest.json')
        source = timeline['source']
        jobs = restore_jobs(timeline, manifest, args.jobs, args.max_review_sec)
        features = read_json(run/'audio_events.json')
        records, local_signature = (load_records(run, source)
            if any(j['phase'] == 'review' for j in jobs) else (None, None))
        cfg = read_json(ROOT/'llm_api.json')['llm']
        parsed = urlsplit(cfg['base_url'])
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or
                parsed.password or parsed.query or parsed.fragment or
                not isinstance(cfg['api_key'], str) or not cfg['api_key']):
            raise ValueError('Invalid provider configuration')
    except (OSError, ValueError, KeyError, TypeError):
        parser.error('Unable to load valid replay jobs, local evidence or provider configuration')
    settings = {'endpoint': cfg['base_url'], 'key': cfg['api_key'], 'model': args.model,
        'timeout': args.timeout, 'retries': args.retries, 'frame_width': args.width,
        'reasoning_effort': args.reasoning_effort, 'max_tokens': args.max_tokens,
        'coarse_fps': 2, 'review_fps': 8, 'modality': 'frames',
        'fresh_run': time.time_ns(), 'local_signature': local_signature}
    frame_cache = run/'event_frames'
    if frame_cache.is_dir():
        settings['frame_cache'] = str(frame_cache)
    # Reserve atomically only after validating all input; never reuse old caches.
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output/'event_cache'
    began = time.monotonic()
    recorded_ids = {request['job']['id'] for request in timeline.get('requests', [])}
    context_origins = {job['id']: ('recorded_request' if job['id'] in recorded_ids else
        'default_24_4_reconstruction' if job['phase'] == 'coarse' else
        'manifest_rally_reconstruction') for job in jobs}
    report = {'run': str(run), 'source': source['identity'], 'model': args.model,
        'endpoint': cfg['base_url'], 'audio_uploaded': False, 'semantic_cache_reused': False,
        'replay_scope': {'same_source_and_context': True if all(
                origin == 'recorded_request' for origin in context_origins.values()) else None,
            'byte_identical_payload': False,
            'context_origins': context_origins,
            'context_audit': 'Without a recorded request, coarse contexts use 24/4 defaults and '
                'review contexts use the current saved manifest. These reconstructions do not '
                'prove identical contexts or payload bytes for arbitrary historical configurations.',
            'coarse_sound_evidence': 'Uses saved full-source audio_events and its global sound IDs, '
                'filtered to the context by production code. Original coarse calls used '
                'chunk-local sound lists and IDs; this is not a byte-for-byte request replay.'},
        'settings': {k: v for k, v in settings.items() if k not in ('key', 'endpoint', 'frame_cache')},
        'code_sha256': {name: digest(APP/name) for name in
                        ('semantic.py', 'llm_transport.py', 'events.py', 'event_schema.py',
                         'prompts/understand_events.md')},
        'status': 'running', 'results': [],
        'note': 'A valid empty or nonempty response does not establish recognition accuracy; '
                'failed attempts are null when the production exception has no attempt count.'}
    save_json(args.output/'summary.json', report)

    def replay(job):
        started = time.monotonic()
        result = {'job': job, 'context_origin': context_origins[job['id']],
                  'status': 'failed', 'attempts': None, 'event_count': None}
        # Each task owns its clock, including finite request retries and local work.
        deadline = started + 120 + args.timeout*(args.retries+1) + sum(2**i for i in range(args.retries))
        try:
            value = understand_context(job, source, cache, settings, None, features,
                deadline, records if job['phase'] == 'review' else None)
            result.update(status='valid', attempts=value['attempts'],
                event_count=len(value['events']), request=safe_request(value.get('request', {})),
                cached=value.get('cached', False), signature=value.get('signature'))
            if 'wire_protocol' in value:
                result['wire_protocol'] = (value['wire_protocol'] if
                    value['wire_protocol'] in ('frame_anchors_v2', 'frame_anchors_v3') else 'other')
            if isinstance(value.get('dimension_encodings'), list):
                result['dimension_encodings'] = [encoding if encoding in ('named_entries', 'named_object')
                    else 'other' for encoding in value['dimension_encodings']]
            if 'retry_history' in value:
                result['retry_history'] = safe_retry_history(value['retry_history'])
        except Exception as exc:
            result.update(failure_diagnostics(exc))
        result['elapsed_sec'] = round(time.monotonic()-started, 3)
        save_json(args.output/f'{job["id"]}.json', result)
        return result

    results = {}
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(jobs))) as pool:
        for future in as_completed([pool.submit(replay, job) for job in jobs]):
            result = future.result()
            results[result['job']['id']] = result
            report['results'] = [results[j['id']] for j in jobs if j['id'] in results]
            report['elapsed_sec'] = round(time.monotonic()-began, 3)
            save_json(args.output/'summary.json', report)
            print(json.dumps({**{k: result[k] for k in ('status', 'elapsed_sec', 'attempts', 'event_count')},
                'job': result['job']['id'], 'error': result.get('error')}, ensure_ascii=False), flush=True)
    report['status'] = 'completed' if all(r['status'] == 'valid' for r in report['results']) else 'completed_with_failures'
    report['elapsed_sec'] = round(time.monotonic()-began, 3)
    save_json(args.output/'summary.json', report)
    return report


def cli_main(argv=None):
    """Keep diagnostic reports, but fail automation when any replay failed."""
    report = main(argv)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(cli_main())
