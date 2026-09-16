#!/usr/bin/env python3
"""Compare advertised models on identical, explicitly chosen local video contexts.

Uploads sampled pictures to the configured provider; no audio is uploaded.
Uses the production prompt and validator, never treats an empty reply as accuracy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volleymole.common import APP, digest, read_json, save_json
from volleymole.event_schema import event_request_payload, decode_event_response
from volleymole.semantic import ResponseContractError, request_json, sampled_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing single-source analysis directory')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--reasoning-effort', choices=('low', 'high', 'max'),
                        help='Optional provider reasoning setting; omitted by default')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Use a new output directory to retain earlier measurements')
    if not 64 <= args.width <= 4096 or not 256 <= args.max_tokens <= 16384:
        parser.error('Width must be 64–4096; max-tokens must be 256–16384')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Timeout must be finite and positive')
    manifest = read_json(args.run/'match_manifest.json')
    source = manifest['source']
    if not 0 <= args.start < args.end <= source['duration_sec']:
        parser.error('Context must lie within the source')
    cfg = read_json(ROOT/'llm_api.json')['llm']
    with urlopen(Request(cfg['base_url'].rstrip('/')+'/models',
            headers={'Authorization': 'Bearer '+cfg['api_key']}), timeout=30) as response:
        advertised = [m['id'] for m in json.load(response)['data']]
    if set(args.models)-set(advertised):
        parser.error('Requested model is not advertised by the configured provider')
    frame_cache = args.run/'event_frames'
    prompt = (APP/'prompts/understand_events.md').read_text(encoding='utf-8')
    payloads = {}
    for phase, fps in (('coarse', 2), ('review', 8)):
        evidence, content, frames = sampled_evidence(source, args.start, args.end, fps,
            args.width, deadline=time.monotonic()+60,
            frame_cache=frame_cache if frame_cache.is_dir() else None)
        content.insert(0, {'type': 'text', 'text': json.dumps({'phase': phase,
            'context': [args.start, args.end], 'requested_fps': fps,
            'actual_sample_times': [t for t, _ in frames], 'candidate': None,
            'local': {'audio_status': 'unknown', 'sound_events': [], 'audio_windows': [],
                      'motion': [], 'action_hypotheses': [], 'action_model_status': 'not_configured'},
            'evidence': evidence}, ensure_ascii=False)})
        payloads[phase] = (evidence, event_request_payload('', prompt, content, evidence,
            args.start, args.end, args.max_tokens, args.reasoning_effort))

    def check(item):
        index, model, phase = item
        evidence, payload = payloads[phase]
        began = time.monotonic()
        result = {'model': model, 'phase': phase, 'frame_count': len(evidence),
            'context': [args.start, args.end], 'width': args.width, 'max_tokens': args.max_tokens,
            'reasoning_effort': args.reasoning_effort,
            'payload_bytes': len(json.dumps(payload).encode())}
        try:
            data, metadata = request_json(cfg['base_url'], cfg['api_key'],
                {**payload, 'model': model}, args.timeout)
            result['request'] = metadata
            # Retain parsed output even when strict validation rejects it.
            result['response'] = data
            events = decode_event_response(data, evidence, args.start, args.end)['events']
            result.update(status='valid', event_count=len(events))
        except Exception as exc:
            result.update(status='failed', error=type(exc).__name__)
            if isinstance(exc, HTTPError):
                result['http_status'] = exc.code
                # Provider errors can echo inputs. Record only known diagnostic categories.
                message = exc.read(8192).decode(errors='replace').lower()
                result['context_limit_reported'] = any(s in message for s in
                    ('context length', 'context window', 'maximum context', 'max_model_len'))
                # Extract only numeric capacity diagnostics, never echoed input.
                for field, pattern in {
                    'context_limit_tokens': r'maximum context length is\s+(\d+)',
                    'input_tokens_reported': r'(?:request has|requested)\s+(\d+)\s+input tokens',
                    'requested_tokens_reported': r'requested\s+(\d+)\s+tokens',
                }.items():
                    match = re.search(pattern, message)
                    if match:
                        result[field] = int(match.group(1))
                exc.close()
            elif isinstance(exc, ResponseContractError):
                result.update(response_contract=exc.reason, finish_reason=exc.finish_reason)
            elif isinstance(exc, ValueError):
                result['validation_error'] = str(exc)[:200]
        result['elapsed_sec'] = round(time.monotonic()-began, 3)
        save_json(args.output/f'probe_{index:02d}.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('response', 'request')}, ensure_ascii=False), flush=True)
        return result

    jobs = [(i, model, phase) for i, (model, phase) in enumerate(
        ( (model, phase) for phase in payloads for model in args.models))]
    with ThreadPoolExecutor(max_workers=min(3, len(jobs))) as pool:
        results = list(pool.map(check, jobs))
    save_json(args.output/'report.json', {'endpoint': cfg['base_url'], 'advertised_models': advertised,
        'source': source['identity'], 'prompt_sha256': digest(APP/'prompts/understand_events.md'),
        'audio_uploaded': False, 'results': results,
        'note': 'Same source interval at 2/8 FPS. Valid JSON and nonempty predictions do not establish recognition accuracy.'})


if __name__ == '__main__':
    main()
