#!/usr/bin/env python3
"""Render model-only blooper observations; never create human scoring forms."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def checked_source(path, expected, cache):
    path = Path(path).resolve()
    if path not in cache:
        if not path.is_file():
            raise ValueError('Source video is missing')
        actual = sha256(path)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
            '-show_streams', '-of', 'json', str(path)], text=True))
        cache[path] = (actual, float(probe['format']['duration']))
    actual, duration = cache[path]
    if not expected or expected != actual:
        raise ValueError('Source SHA256 is missing or differs from the annotation source')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Source duration is invalid')
    return path, duration


def render(input_path, output):
    document = json.loads(input_path.read_text(encoding='utf-8'))
    failure_path = input_path.parent/'failures.json'
    annotation_failures = json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.is_file() else []
    rows = document.get('items', document.get('annotations', document.get('results', [])))
    if not isinstance(rows, list):
        raise ValueError('annotations must be an array')
    sources = document.get('sources', {})
    if isinstance(sources, list):
        sources = {str(s.get('source_id', s.get('id', i))): s for i, s in enumerate(sources)}
    cache, cards, export, errors = {}, [], [], []
    esc = lambda value: html.escape(str(value), quote=True)
    for index, row in enumerate(rows):
        source = sources.get(str(row.get('source_id', 'single')), {})
        if isinstance(row.get('source'), dict):
            source = {**source, **row['source']}
        try:
            if row.get('status', 'complete') != 'complete':
                raise ValueError('Annotation did not complete: '+str(row.get('status')))
            path = row.get('source_path', source.get('path', source.get('source_path')))
            expected = row.get('source_sha256', source.get('sha256', source.get('source_sha256')))
            path, duration = checked_source(path, expected, cache)
            start = row.get('start_sec', row.get('clip_start_sec'))
            end = row.get('end_sec', row.get('clip_end_sec'))
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= duration+.001):
                raise ValueError('Annotation interval exceeds verified source duration')
            annotation = row.get('annotation', row.get('labels', row.get('result', {})))
            if not isinstance(annotation, dict):
                raise ValueError('Annotation payload is missing')
            ratings = annotation.get('ratings', annotation.get('dimensions', {}))
            flags = annotation.get('flags', {})
            facts = annotation.get('observed_facts', annotation.get('observations', annotation.get('facts')))
            unknown = annotation.get('unknown_reason', annotation.get('uncertainty'))
            stages = annotation.get('stages', {})
            for stage, span in stages.items():
                a, b = span['start_sec'], span['end_sec']
                if a is None and b is None:
                    continue
                if (type(a) not in (int, float) or type(b) not in (int, float)
                        or not math.isfinite(a) or not math.isfinite(b) or not start <= a <= b <= end):
                    raise ValueError('Stage interval exceeds verified annotation window: '+stage)
            src = path.as_uri()+f'#t={start:.6f},{end:.6f}'
            def section(title, value):
                return '<h3>'+esc(title)+'</h3><pre>'+esc(json.dumps(value, ensure_ascii=False, indent=2))+'</pre>'
            cards.append('<article><h2>片段 '+str(index+1)+' · '+f'{start:.2f}–{end:.2f} 秒</h2>'
                +'<video controls preload="metadata" src="'+esc(src)+'" data-start="'+str(start)+'" data-end="'+str(end)+'"></video>'
                +section('自动观察', facts)+section('自动评分（不是人工真值）', ratings)
                +section('状态', flags)+section('三阶段（原视频秒数）', stages)+section('不确定性', unknown)+'</article>')
            export.append({'index': index+1, 'source_path': str(path), 'source_sha256': expected,
                'start_sec': start, 'end_sec': end, 'annotation_origin': 'model_auto_annotation_not_ground_truth',
                'facts': json.dumps(facts, ensure_ascii=False), 'ratings': json.dumps(ratings, ensure_ascii=False),
                'flags': json.dumps(flags, ensure_ascii=False), 'stages': json.dumps(stages, ensure_ascii=False),
                'uncertainty': json.dumps(unknown, ensure_ascii=False)})
        except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            errors.append({'index': index+1, 'error': str(exc)})
            cards.append('<article><h2>片段 '+str(index+1)+' · 无法展示</h2><p>'+esc(exc)+'</p></article>')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'annotations.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['index', 'source_path', 'source_sha256', 'start_sec', 'end_sec', 'annotation_origin',
                  'facts', 'ratings', 'flags', 'stages', 'uncertainty']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(export)
    markup = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>比赛自动标注</title><style>body{background:#f1f4f7;color:#17202a;font:16px/1.6 system-ui;margin:0}main{max-width:1000px;margin:auto;padding:24px}article{background:white;padding:20px;margin:20px 0;border-radius:8px}video{width:100%;max-height:60vh;background:#111}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:12px}h3{font-size:16px;margin-bottom:0}.notice{padding:14px;background:#fff0cc;border-radius:6px}</style>
<main><h1>比赛自动标注</h1><p class="notice">模型自动标注 · 可能有错 · 不是真值。未写入人工评分表。</p>'''
    markup += '<p>读取 '+str(len(rows))+' 条，验证通过 '+str(len(export))+' 条，错误 '+str(len(errors))+' 条。原始音轨随原视频播放；仅视觉标注中的笑声仍为未知。</p>'
    if annotation_failures:
        markup += '<p>另有 '+str(len(annotation_failures))+' 个请求未完成标注。</p><details><summary>未完成请求</summary><pre>'+esc(json.dumps(annotation_failures, ensure_ascii=False, indent=2))+'</pre></details>'
    markup += ''.join(cards)+'''</main><script>
document.querySelectorAll('video').forEach(v=>{const a=Number(v.dataset.start),b=Number(v.dataset.end);v.addEventListener('loadedmetadata',()=>{v.currentTime=a;});v.addEventListener('play',()=>{if(v.currentTime<a||v.currentTime>=b)v.currentTime=a;});v.addEventListener('timeupdate',()=>{if(v.currentTime>=b)v.pause();});});
</script></html>'''
    (output/'index.html').write_text(markup, encoding='utf-8')
    report = {'input_sha256': sha256(input_path), 'annotations': len(rows), 'rendered': len(export),
              'errors': errors, 'annotation_failures': annotation_failures,
              'origin': 'model_auto_annotation_not_ground_truth',
              'media_policy': 'Verified local source URI with bounded playback; original audio preserved.'}
    (output/'render_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='annotations.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
