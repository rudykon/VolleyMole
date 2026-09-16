"""Bounded, strict JSON transport for the OpenAI-compatible chat endpoint.

Retries deliberately belong to the caller: it owns the phase deadline and the
attempt budget. This module never repairs JSON or turns a failed response into
an empty result, and its exceptions never retain remote response text.
"""
import hashlib
from http.client import HTTPException
import json
import math
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_FINISH_REASONS = ('stop', 'length', 'content_filter', 'tool_calls', None)
_DEGRADED_REASONS = (
    'model_unsupported', 'json_object_unsupported_on_anthropic',
    'json_schema_missing_schema', 'schema_keywords_stripped',
)
_JSON_FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', re.DOTALL | re.IGNORECASE)


class ResponseContractError(ValueError):
    """Safe diagnostics without retaining remote response bodies or credentials."""

    def __init__(self, reason, finish_reason=None):
        super().__init__(reason)
        self.reason = reason
        self.finish_reason = finish_reason if finish_reason in _FINISH_REASONS else 'other'


class TransportError(OSError):
    """A redacted transport failure with an explicit, finite retry policy."""

    def __init__(self, reason, *, retryable, http_status=None):
        super().__init__(reason)
        self.reason = reason
        self.retryable = bool(retryable)
        self.http_status = http_status


def is_retryable_transport_error(error):
    """Also recognize raw exceptions injected by tests or alternate openers."""
    if isinstance(error, TransportError):
        return error.retryable
    if isinstance(error, HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (TimeoutError, URLError, ConnectionError, HTTPException))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite_json_number')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite_json_number')
    return result


def _strict_json(text, reason, finish_reason=None):
    try:
        # JSONDecoder otherwise accepts duplicate keys and NaN/Infinity.
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ResponseContractError(reason, finish_reason) from None


def parse_json_content(content, finish_reason='stop'):
    """Accept exactly one JSON value, optionally inside one complete JSON fence.

    No substring extraction, brace balancing, trailing-comma repair, prose
    removal, or schema coercion is permitted. Domain validation remains the
    caller's responsibility.
    """
    if not isinstance(content, str):
        raise ResponseContractError('invalid_response_envelope', finish_reason)
    text = content.strip()
    match = _JSON_FENCE.fullmatch(text)
    encoding = 'fenced_json' if match else 'json'
    if match:
        text = match.group(1)
    return _strict_json(text, 'invalid_json', finish_reason), encoding


def _safe_usage(usage):
    """Only persist recognized numeric token counters, never arbitrary strings."""
    if not isinstance(usage, dict):
        return None
    allowed = {
        'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None,
        'prompt_tokens_details': ('cached_tokens', 'audio_tokens', 'image_tokens', 'video_tokens'),
        'completion_tokens_details': ('reasoning_tokens', 'audio_tokens',
                                      'accepted_prediction_tokens', 'rejected_prediction_tokens'),
    }
    def counter(value):
        return type(value) is int and value >= 0
    safe = {}
    for key, children in allowed.items():
        value = usage.get(key)
        if children is None:
            if counter(value):
                safe[key] = value
        elif isinstance(value, dict):
            safe[key] = {name: value[name] for name in children if counter(value.get(name))}
    return safe


def _read_response(response, deadline):
    chunks, size = [], 0
    reader = getattr(response, 'read1', None) or response.read
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('semantic response deadline')
        sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
        if sock is not None:
            sock.settimeout(remaining)
        chunk = reader(min(65536, MAX_RESPONSE_BYTES + 1 - size))
        if time.monotonic() >= deadline:
            raise TimeoutError('semantic response deadline')
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise ResponseContractError('response_too_large')
        chunks.append(chunk)
    return b''.join(chunks)


def request_json(endpoint, key, payload, timeout, *, opener=None, preserve_http_error=False):
    """Make one request with bounded response size/time and strict envelopes.

    ``opener`` is injectable for a legacy wrapper or deterministic tests. The
    deadline starts before opening/uploading; every subsequent read gets only
    the time remaining, not a fresh full socket timeout. urllib's connection
    setup uses socket-operation timeouts, not a cancellable wall-clock limit.
    ``preserve_http_error`` keeps the legacy HTTPError/body ownership contract
    for callers that already classify provider failures; those callers must
    sanitize diagnostics and close the error response themselves.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    deadline = time.monotonic() + timeout
    request = Request(endpoint.rstrip('/') + '/chat/completions',
                      data=json.dumps(payload, allow_nan=False).encode(),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TransportError('timeout', retryable=True)
    try:
        with (opener or urlopen)(request, timeout=remaining) as response:
            degraded = getattr(response, 'headers', {}).get('X-Structured-Output-Degraded')
            if degraded is not None and degraded not in _DEGRADED_REASONS:
                degraded = 'other'
            # AiHubmix documents this exact value when its opt-in, key-level
            # syntax repair ran. This observes the header, not the key setting.
            repaired = getattr(response, 'headers', {}).get('X-JSON-Repaired') == 'true'
            raw = _read_response(response, deadline)
    except HTTPError as exc:
        if preserve_http_error:
            raise
        status = exc.code
        exc.close()
        raise TransportError('http_error', retryable=status in (408, 429) or 500 <= status < 600,
                             http_status=status) from None
    except TimeoutError:
        raise TransportError('timeout', retryable=True) from None
    except (URLError, ConnectionError, HTTPException):
        # RemoteDisconnected and IncompleteRead are not reliably URLError.
        raise TransportError('connection_error', retryable=True) from None
    except OSError:
        raise TransportError('io_error', retryable=False) from None

    data = _strict_json(raw, 'invalid_response_envelope')
    if not isinstance(data, dict) or not isinstance(data.get('choices'), list) or len(data['choices']) != 1:
        raise ResponseContractError('invalid_response_envelope')
    choice = data['choices'][0]
    if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
        raise ResponseContractError('invalid_response_envelope')
    finish = choice.get('finish_reason')
    message = choice['message']
    if message.get('refusal'):
        raise ResponseContractError('refused', finish)
    if finish != 'stop':
        raise ResponseContractError('incomplete_response', finish)
    parsed, encoding = parse_json_content(message.get('content'), finish)
    format_ = payload.get('response_format', {})
    metadata = {
        'model': payload['model'], 'finish_reason': finish, 'usage': _safe_usage(data.get('usage')),
        'response_format': format_.get('type'), 'structured_output_degraded': degraded,
        'gateway_json_repaired': repaired,
        'reasoning_effort': payload.get('reasoning_effort'), 'response_content_encoding': encoding,
    }
    if format_.get('type') == 'json_schema':
        metadata['response_schema_sha256'] = hashlib.sha256(
            json.dumps(format_['json_schema'], sort_keys=True).encode()).hexdigest()
    return parsed, metadata
