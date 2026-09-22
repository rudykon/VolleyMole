"""Optional, offline audio cues on an approved edit; video packets are copied.

Cue selection is supplied by an editorial or API-reviewed plan with local assets.
This module does not infer an action, fetch media, or call a model.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np

from .common import digest, read_json, save_json

RATE = 48000
CHANNELS = 2


def media_info(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))


def video_signature(path):
    """Check both encoded picture content and the entire packet timeline."""
    content = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
        '-c:v', 'copy', '-f', 'streamhash', '-hash', 'sha256', '-']).decode().strip()
    packets = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts_time,dts_time,duration_time,size,flags',
        '-of', 'json', str(path)]))['packets']
    return {'encoded_video': content, 'packets': len(packets),
            'timeline_sha256': hashlib.sha256(json.dumps(packets, sort_keys=True).encode()).hexdigest()}


def finite_number(value, field, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{field} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{field} must be between {minimum} and {maximum}')
    return value


def validate_plan(plan, directory, duration):
    if plan.get('version') != 1:
        raise ValueError('Unsupported audio cue plan version')
    cues = plan.get('cues')
    maximum = plan.get('max_cues', 2)
    if type(maximum) is not int or not 1 <= maximum <= 20:
        raise ValueError('max_cues must be an integer in 1..20')
    if not isinstance(cues, list) or not 0 <= len(cues) <= maximum:
        raise ValueError('Cue count exceeds the plan density limit')
    result = []
    ids = set()
    for cue in cues:
        name = cue.get('id')
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError('Each cue needs a unique nonempty id')
        ids.add(name)
        if not isinstance(cue.get('asset'), str):
            raise ValueError('Cue asset must be a local file path')
        asset = (Path(directory) / cue['asset']).resolve()
        if not asset.is_file():
            raise ValueError(f'Missing local audio asset: {asset}')
        start = finite_number(cue.get('source_start_sec', 0), 'source_start_sec', 0, 86400)
        end = finite_number(cue.get('source_end_sec'), 'source_end_sec', start, start + 5)
        length = end - start
        if length < .1:
            raise ValueError('Audio cues must last 0.1..5 seconds')
        at = finite_number(cue.get('at_sec'), 'at_sec', 0, duration)
        if at + length > duration + .5 / RATE:
            raise ValueError('Audio cue would be truncated at the end of the video')
        streams = media_info(asset)['streams']
        audio = next((s for s in streams if s['codec_type'] == 'audio'), None)
        if audio is None:
            raise ValueError(f'Asset has no audio: {asset}')
        if end > float(audio.get('duration', end)) + .001:
            raise ValueError(f'Audio cue exceeds source duration: {name}')
        result.append({**cue, 'asset': str(asset), 'at_sec': at,
                       'source_start_sec': start, 'source_end_sec': end,
                       'rms_db': finite_number(cue.get('rms_db', -20), 'rms_db', -36, -14),
                       'duck_db': finite_number(cue.get('duck_db', -3), 'duck_db', -12, 0)})
    result.sort(key=lambda c: c['at_sec'])
    for left, right in zip(result, result[1:]):
        if left['at_sec'] + left['source_end_sec'] - left['source_start_sec'] > right['at_sec']:
            raise ValueError('Audio cues must not overlap')
    return result


def decode_audio(path, target, *, start=None, duration=None, samples=None):
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(path)]
    if start is not None:
        command += ['-ss', str(start), '-t', str(duration)]
    command += ['-map', '0:a:0', '-vn', '-ar', str(RATE), '-ac', str(CHANNELS)]
    if samples is not None:
        command += ['-af', f'aresample={RATE}:async=1:first_pts=0,apad,atrim=end_sample={samples}']
    command += ['-c:a', 'pcm_f32le', '-f', 'f32le', str(target)]
    subprocess.run(command, check=True)


def mix_cue(background, sound, rms_db, duck_db):
    """Return a bounded cue mix; original PCM outside this slice is untouched."""
    rms = float(np.sqrt(np.mean(sound.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(sound)))
    if not math.isfinite(rms) or rms < 1e-6:
        raise ValueError('Audio cue is silent or invalid')
    gain = min(10 ** (rms_db / 20) / rms, 10 ** (-6 / 20) / peak)
    envelope = np.ones((len(sound), 1), dtype=np.float32)
    attack = min(round(.025 * RATE), len(sound) // 3)
    release = min(round(.080 * RATE), len(sound) // 3)
    envelope[:attack, 0] = np.linspace(0, 1, attack)
    envelope[-release:, 0] = np.linspace(1, 0, release)
    bed = background * (1 - envelope * (1 - 10 ** (duck_db / 20)))
    addition = sound * gain * envelope
    # Lower the entire cue's gain if needed; no clipping or global limiter.
    ceiling = 10 ** (-1 / 20)
    positive = addition > 1e-9
    negative = addition < -1e-9
    scale = 1.
    if np.any(positive):
        scale = min(scale, float(np.min((ceiling - bed[positive]) / addition[positive])))
    if np.any(negative):
        scale = min(scale, float(np.min((-ceiling - bed[negative]) / addition[negative])))
    if scale <= 0 or float(np.max(np.abs(bed))) > ceiling:
        raise ValueError('Source is too loud at this cue; choose another point or lower the source first')
    mixed = bed + addition * scale
    return mixed, {'asset_gain_db': 20 * math.log10(gain * scale),
                   'additional_headroom_reduction_db': 20 * math.log10(scale),
                   'mixed_peak_dbfs': 20 * math.log10(max(float(np.max(np.abs(mixed))), 1e-12))}


def apply_audio_settings(plan, settings):
    from .templates import validate_audio
    validate_audio(settings)
    result = {**plan, 'cues': [dict(cue) for cue in plan['cues']]}
    if 'max_cues' in settings:
        result['max_cues'] = settings['max_cues']
    if settings.get('enabled') is False:
        result['cues'] = []
    for cue in result['cues']:
        cue.update({key: settings[key] for key in ('rms_db', 'duck_db') if key in settings})
    return result


def render(video, plan_path, output, *, audio_settings=None):
    video, plan_path, output = map(lambda p: Path(p).resolve(), (video, plan_path, output))
    report_path = output.with_suffix('.audio.json')
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must be a new .mp4 file')
    if output.exists() or report_path.exists():
        raise ValueError('Output or audio report already exists; use a new output name')
    streams = media_info(video)['streams']
    picture = next(s for s in streams if s['codec_type'] == 'video')
    sound = next((s for s in streams if s['codec_type'] == 'audio'), None)
    if sound is None:
        raise ValueError('The approved video must contain its original audio')
    if abs(float(picture.get('start_time', 0))) > .001 or abs(float(sound.get('start_time', 0))) > .001:
        raise ValueError('Input video and audio must start at time zero')
    duration = float(int(picture['duration_ts']) * Fraction(picture['time_base']))
    plan = read_json(plan_path)
    if audio_settings is not None:
        plan = apply_audio_settings(plan, audio_settings)
    cues = validate_plan(plan, plan_path.parent, duration)
    samples = round(duration * RATE)
    signature = video_signature(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cues:
        report = {'version': 1, 'source': str(video), 'source_sha256': digest(video),
                  'plan': str(plan_path), 'plan_sha256': digest(plan_path),
                  'output': str(output), 'output_sha256': digest(video),
                  'video_unchanged': True, 'video_signature': signature,
                  'duration_sec': duration, 'cue_count': 0, 'cues': [],
                  'audio_mode': 'original_unchanged', 'audio_settings': audio_settings, 'new_model_calls': 0,
                  'selection_mode': plan.get('selection_mode', 'explicit_editorial_times')}
        # A legitimate model abstention preserves both original audio and video.
        with tempfile.TemporaryDirectory(prefix='meme-audio-', dir=output.parent) as temp:
            candidate = Path(temp) / 'unchanged.mp4'
            shutil.copyfile(video, candidate)
            candidate.replace(output)
        save_json(report_path, report)
        return report
    with tempfile.TemporaryDirectory(prefix='meme-audio-', dir=output.parent) as temp:
        temp = Path(temp)
        pcm = temp / 'mix.f32'
        decode_audio(video, pcm, samples=samples)
        if pcm.stat().st_size != samples * CHANNELS * 4:
            raise RuntimeError('Decoded audio does not match the picture duration')
        mixed = np.memmap(pcm, mode='r+', dtype='<f4', shape=(samples, CHANNELS))
        observations = []
        for index, cue in enumerate(cues):
            length = cue['source_end_sec'] - cue['source_start_sec']
            expected = round(length * RATE)
            source = temp / f'cue-{index}.f32'
            decode_audio(cue['asset'], source, start=cue['source_start_sec'], duration=length)
            audio = np.fromfile(source, dtype='<f4').reshape(-1, CHANNELS)
            if len(audio) != expected:
                raise ValueError(f'Audio cue decoded with unexpected length: {cue["id"]}')
            begin = round(cue['at_sec'] * RATE)
            end = begin + expected
            if end > samples:
                raise ValueError('Rounded audio cue exceeds the picture duration')
            addition, stats = mix_cue(mixed[begin:end], audio, cue['rms_db'], cue['duck_db'])
            mixed[begin:end] = addition
            observations.append({**cue, **stats, 'start_sample': begin, 'end_sample': end,
                                 'asset_sha256': digest(cue['asset'])})
        mixed.flush()
        del mixed
        candidate = temp / 'edition.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(video),
                        '-f', 'f32le', '-ar', str(RATE), '-ac', str(CHANNELS), '-i', str(pcm),
                        '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '0',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                        '-movflags', '+faststart', str(candidate)], check=True)
        if video_signature(candidate) != signature:
            raise RuntimeError('Encoded picture content or timing changed')
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(candidate),
                        '-f', 'null', '-'], check=True)
        report = {'version': 1, 'source': str(video), 'source_sha256': digest(video),
                  'plan': str(plan_path), 'plan_sha256': digest(plan_path),
                  'output': str(output), 'output_sha256': digest(candidate),
                  'video_unchanged': True, 'video_signature': signature,
                  'duration_sec': duration, 'cue_count': len(cues), 'cues': observations,
                  'audio_mode': 'local_cues_with_short_ducking', 'audio_settings': audio_settings, 'new_model_calls': 0,
                  'selection_mode': plan.get('selection_mode', 'explicit_editorial_times')}
        candidate.replace(output)
        save_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='给已有成片添加克制的本地梗配音，画面与时间线原样保留')
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--template', help='读取与剪辑共用的模板 audio 设置')
    parser.add_argument('--max-cues', type=int, help='配音数量上限；覆盖模板')
    parser.add_argument('--rms-db', type=float, help='配音电平，-36 至 -14 dB；覆盖模板及计划')
    parser.add_argument('--duck-db', type=float, help='配音期间原声增益，-12 至 0 dB；覆盖模板及计划')
    parser.add_argument('--audio-enabled', action=argparse.BooleanOptionalAction, default=None,
                        help='启用配音；--no-audio-enabled 保留完整原声')
    args = parser.parse_args(argv)
    from .templates import load_template
    settings = dict(load_template(args.template).get('audio', {})) if args.template else {}
    for key in ('max_cues', 'rms_db', 'duck_db'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.audio_enabled is not None:
        settings['enabled'] = args.audio_enabled
    result = render(args.video, args.plan, args.output, audio_settings=settings or None)
    print(json.dumps({'output': result['output'], 'cue_count': result['cue_count'],
                      'video_unchanged': result['video_unchanged']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
