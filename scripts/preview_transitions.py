"""Render five transition demos and optional reusable ProRes 4444 overlays."""
import argparse
from html import escape
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from volleymole.common import DEFAULT_FONT, probe, save_json
from volleymole.illustrated import encode_transition, title_card
from volleymole.media_worker import concatenate_segments
from volleymole.presentation import lively_title
from volleymole.transitions import STYLE_IDS, STYLE_LABELS, TransitionRenderer

PAIRINGS = {'velocity': ('manga', 'arena-zh'), 'paper': ('papercut', 'editorial-zh'),
            'ink': ('ink', 'cinema-zh'), 'prism': ('clay', 'minimal-en'), 'film': ('retro', 'cinema-en')}


def encode_frames(path, frames, count, alpha=False, quality='720p'):
    from volleymole.quality import get_quality
    q=get_quality(quality)
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra' if alpha else 'bgr24',
               '-s', '720x1280', '-r', '30', '-i', 'pipe:0']
    if alpha:
        command += ['-an', '-c:v', 'prores_ks', '-profile:v', '4', '-pix_fmt', 'yuva444p10le', '-alpha_bits', '16']
    else:
        command += ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-c:v', 'libx264',
                    '-preset', 'fast', '-crf', str(20 if quality=='720p' else q.crf), '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-t', str(count/30)]
    if quality!='720p':command += ['-vf',f'scale={q.width}:{q.height}:flags=lanczos']
    command += ['-threads', '4', '-frames:v', str(count), str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames: process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait(): raise RuntimeError(f'编码失败：{path}')
    finally:
        if process.poll() is None: process.terminate(); process.wait()


def decode(path, alpha=False, normalize=False):
    command = ['ffmpeg', '-v', 'error', '-xerror', '-threads', '2', '-i', str(path),
               *(['-vf','scale=720:1280'] if normalize else []),
               '-f', 'rawvideo', '-pix_fmt', 'bgra' if alpha else 'bgr24', '-threads', '2', 'pipe:1']
    data = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(data, np.uint8).reshape(-1, 1280, 720, 4 if alpha else 3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/transition-templates'))
    parser.add_argument('--before', type=Path, help='可选：前一段已渲染 720×1280、30 fps、带音轨视频')
    parser.add_argument('--after', type=Path, help='可选：后一段已渲染视频；必须与 --before 同时提供')
    parser.add_argument('--title', help='示例原始标题；使用真实视频时请填写其剪辑单标题')
    parser.add_argument('--export-overlays', action='store_true', help='另导出 0.8 秒透明 ProRes 4444 MOV，正向及反向各一段')
    args = parser.parse_args(argv)
    if bool(args.before) != bool(args.after): parser.error('--before 和 --after 必须同时提供')
    args.output.mkdir(parents=True, exist_ok=False)
    item = {'rank': 1, 'title': args.title or ('球场时刻' if args.before else '长回合起跳对抗与防守站位')}
    rows, html_rows = [], []
    external = None
    if args.before:
        external = []
        for path in (args.before, args.after):
            meta = probe(path)
            if (meta['width'], meta['height']) != (720, 1280): raise ValueError('预览输入必须已渲染为 720×1280')
            if meta['average_fps'] not in ('30/1','60/2') or not meta['has_audio']:
                raise ValueError('预览输入必须为 30 fps 且含音轨')
            if not 0 < meta['duration_sec'] <= 10:
                raise ValueError('预览仅接受 10 秒以内的短样片；完整回合请使用正式渲染入口')
            external.append({'path': str(path.resolve()), 'duration_sec': meta['duration_sec'], 'output_frames': round(meta['duration_sec']*30)})
    for style in STYLE_IDS[1:]:
        started = time.perf_counter()
        directory = args.output/style
        directory.mkdir()
        art, template = PAIRINGS[style]
        title = lively_title(item, template)
        if external:
            before, after = external
        else:
            boundaries = []
            for rank, name in ((2, 'before'), (1, 'after')):
                boundary_item = dict(item, rank=rank)
                image = title_card(boundary_item, lively_title(boundary_item, template), DEFAULT_FONT, 5, art, template)
                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                path = directory/f'{name}.mp4'
                encode_frames(path, (frame for _ in range(30)), 30)
                boundaries.append({'path': str(path), 'duration_sec': 1., 'output_frames': 30})
            before, after = boundaries
        segment = {'next_rank': 1, 'top_k': 5, 'original_title': item['title'], 'display_title': title,
                   'duration_sec': 3., 'output_frames': 90, 'transition_style': style, 'path': str(directory/'transition.mp4')}
        encode_transition(before, after, directory/'transition.mp4', segment, DEFAULT_FONT, art, template, style)
        concatenate_segments([before, segment, after], directory/'sample.mp4')
        # Full decode verifies 90 actual frames, not just container duration.
        actual = decode(directory/'transition.mp4')
        if len(actual) != 90: raise AssertionError('转场帧数错误')
        expected = cv2.imread(str(directory/'transition.png'))
        maes = [float(np.abs(actual[i].astype(float)-expected.astype(float)).mean()) for i in (15,45,75)]
        if max(maes) >= 5: raise AssertionError((style, maes))
        # Keep a compact native-code contact sheet of animation frames for review.
        frames = [cv2.resize(actual[i], (180,320), interpolation=cv2.INTER_AREA) for i in (0,3,5,7,11,45,78,81,83,85,87,89)]
        cv2.imwrite(str(directory/'timeline.jpg'), np.vstack((np.hstack(frames[:6]), np.hstack(frames[6:]))))
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(directory/'sample.mp4'),'-f','null','-'],check=True)
        compiled = decode(directory/'sample.mp4')
        start = round(before['duration_sec']*30)
        compiled_maes = [float(np.abs(compiled[start+i].astype(float)-expected.astype(float)).mean()) for i in (15,45,75)]
        if max(compiled_maes) >= 5: raise AssertionError((style, compiled_maes))
        del compiled, actual
        overlays = []
        if args.export_overlays:
            dummy = np.zeros((1280,720,3), np.uint8)
            renderer = TransitionRenderer(style,dummy,dummy,dummy)
            for reverse, name in ((False,'in'),(True,'out')):
                path = directory/f'{style}-{name}-alpha.mov'
                encode_frames(path, (renderer.material_layer(i/23,reverse) for i in range(24)), 24, True)
                decoded = decode(path, True)
                if len(decoded) != 24: raise AssertionError('透明转场帧数错误')
                if decoded[0,:,:,3].max() or decoded[-1,:,:,3].max(): raise AssertionError('透明转场首尾不透明')
                if decoded[11:13,:,:,3].min() < 254: raise AssertionError('透明转场未完全遮住切点')
                overlays.append({'file':path.name,'frames':24,'fps':30,'cut_frame':12,'cut_sec':.4,'alpha':'straight','codec':'ProRes 4444'})
        row = {'style':style,'label':STYLE_LABELS[style],'art_theme':art,'title_template':template,
               'title_card_mae':maes,'compiled_title_card_mae':compiled_maes,'full_decode':'passed','overlays':overlays,'elapsed_sec':round(time.perf_counter()-started,3)}
        rows.append(row)
        save_json(args.output/'validation.json',{'status':'in_progress','scope':'Presentation demos only; no inference or ranking rerun.','samples':rows})
        links = ''.join(f'<a href="{style}/{x["file"]}">{"入场" if "-in-" in x["file"] else "退场"}透明 MOV</a> ' for x in overlays)
        html_rows.append(f'<article><div class="number">0{len(rows)}</div><h2>{escape(STYLE_LABELS[style])}</h2><code>--transition-style {style}</code><video controls loop muted playsinline preload="metadata" poster="{style}/transition.png" src="{style}/sample.mp4"></video><p>{art} / {template}</p><nav>{links}<a href="{style}/timeline.jpg">动画帧时间线</a></nav></article>')
        print(row,flush=True)
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VolleyMole · 五种转场节奏</title><style>body{margin:0;background:#101719;color:#edece5;font:16px system-ui}main{max-width:1220px;margin:auto;padding:64px 28px}h1{font-size:clamp(32px,5vw,58px);letter-spacing:-2px;margin:12px 0 24px}header p{max-width:740px;line-height:1.8;color:#a8b8bb}.kicker,code{color:#91c7b2;letter-spacing:2px;font-size:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:44px 28px;margin-top:56px}article{border-top:1px solid #405054;padding-top:20px}.number{color:#9ca7aa;font-size:12px}h2{font-size:25px;margin:10px 0}code{display:block;letter-spacing:0;margin-bottom:22px}video{width:100%;border-radius:10px;background:#060909}article p{font-size:12px;color:#91a5ab}nav{line-height:2}a{color:#c3dfd2;font-size:13px;margin-right:14px}footer{margin-top:48px;color:#9eafb3;font-size:14px;line-height:1.9}</style><main><header><div class="kicker">VOLLEYMOLE / MOTION MATERIALS</div><h1>五种材质，五种入场方式。</h1><p>Codex 内置生图创作材质，程序驱动遮罩动画。点击播放查看：0.4 秒入场、2.2 秒标题完整停留、0.4 秒退场。可独立组合插画与中英标题。</p></header><div class="grid">'''+''.join(html_rows)+'''</div><footer>透明 MOV：720 × 1280 / 30 fps / ProRes 4444 / 0.8 秒。叠放在两段视频的接缝上，底层切点对齐第 12 帧（0.4 秒）。MOV 是程序输出的透明动画层，原始生图材质为不透明 PNG。浏览器通常不能播放 ProRes，请下载后在支持 Alpha 的剪辑软件中使用。示例用于呈现设计，不代表重新检测、排名或完整比赛验收。</footer></main></html>'''
    (args.output/'index.html').write_text(html,encoding='utf-8')
    save_json(args.output/'validation.json',{'status':'passed','scope':'Presentation demos only; no inference or ranking rerun.','samples':rows})
    print(args.output/'index.html',flush=True)


if __name__ == '__main__': main()
