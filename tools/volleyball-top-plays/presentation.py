"""A source-traceable lively edit: teaser, full rallies, slow replays and wipes."""
import math
from pathlib import Path
import subprocess
import time
from common import read_json, save_json

FPS = 30
REPLAY_SPEED = 2 / 3
ACCENTS = [(255, 198, 70), (85, 231, 199), (255, 115, 142)]
INK = (13, 22, 40)
CREAM = (255, 246, 223)


def lively_title(item):
    title = item['title']
    if '极低' in title: return '贴地救球！太拼了'
    if '低姿' in title: return '压低重心，这球我来！'
    if '低位' in title: return '低位防守，反应拉满！'
    if '二传' in title or '组织' in title: return '配合到位，节奏拉满！'
    if '长回合' in title: return '这回合，够来劲！'
    if '攻防' in title or '往返' in title: return '来回拉锯，接着打！'
    return '好球时刻，一起看！'


def peak_window(item, rally, length=2.0):
    """Use the visually reviewed keyframe, never invent an off-clip highlight."""
    start, end = item['clip_start_sec'], item['clip_end_sec']
    peak = rally['preview_times_sec'][1]
    if not math.isfinite(peak) or not start <= peak <= end:
        peak = (start + end) / 2
    length = min(length, end-start)
    a = min(max(start, peak-length*.6), end-length)
    return {'source_start_sec': a, 'source_end_sec': a+length,
            'peak_sec': peak, 'peak_evidence': 'visually_reviewed_preview_peak',
            'rally_id': item['rally_id'], 'rank': item['rank']}


def build_timeline(decision, manifest):
    by_id = {r['rally_id']: r for r in manifest['rallies']}
    segments = []
    def add(kind, frames, **data):
        segments.append(dict(kind=kind, output_frames=frames, duration_sec=frames/FPS, **data))
    # Five hard cuts at 0.8 seconds: a four-second hook for either top-5 or top-10.
    for item in reversed(decision['selected'][:5]):
        window = peak_window(item, by_id[item['rally_id']], .8)
        add('teaser', 24, playback_rate=1., **window)
    for item in reversed(decision['selected']):
        add('transition', 12, next_rank=item['rank'])
        frames = math.ceil((item['clip_end_sec']-item['clip_start_sec'])*FPS-1e-6)
        add('rally', frames, rank=item['rank'], rally_id=item['rally_id'], playback_rate=1.,
            source_start_sec=item['clip_start_sec'], source_end_sec=item['clip_end_sec'])
        window = peak_window(item, by_id[item['rally_id']])
        frames = math.ceil((window['source_end_sec']-window['source_start_sec'])/REPLAY_SPEED*FPS-1e-6)
        add('replay', frames, playback_rate=REPLAY_SPEED, **window)
    offset = 0
    for i, segment in enumerate(segments):
        segment.update(index=i, timeline_start_frame=offset, timeline_start_sec=offset/FPS)
        offset += segment['output_frames']
        segment['timeline_end_sec'] = offset/FPS
    return segments


def validate_timeline(report, decision):
    segments=report['segments'];offset=0
    expected={r['rank']:r for r in decision['selected']}
    if [s['rank'] for s in segments if s['kind']=='rally']!=list(reversed(expected)):
        raise ValueError('完整回合数量或倒计时顺序错误')
    if [s['rank'] for s in segments if s['kind']=='replay']!=list(reversed(expected)):
        raise ValueError('每个回合必须有且只有一次短回放')
    if [s['kind'] for s in segments[:5]]!=['teaser']*5:
        raise ValueError('片头必须是五次快速切换')
    if sum(s['output_frames'] for s in segments if s['kind']=='teaser')!=120:
        raise ValueError('快切片头必须为4秒')
    for segment in segments:
        if segment['timeline_start_frame']!=offset or abs(segment['timeline_start_sec']-offset/FPS)>1e-6:
            raise ValueError('成片时间线不连续')
        offset+=segment['output_frames']
        if abs(segment['timeline_end_sec']-offset/FPS)>1e-6:raise ValueError('成片结束点不连续')
        if segment['kind']!='transition':
            item=expected[segment['rank']]
            a,b=segment['source_start_sec'],segment['source_end_sec']
            if not item['clip_start_sec']-1e-6<=a<b<=item['clip_end_sec']+1e-6:
                raise ValueError('预告或回放越出原回合')
            if segment['kind']=='rally' and (a!=item['clip_start_sec'] or b!=item['clip_end_sec']):
                raise ValueError('转场不能裁断完整回合')
        if not Path(segment['path']).is_file():raise ValueError('缺失时间线片段')
    if abs(report['expected_duration_sec']-offset/FPS)>1e-6:raise ValueError('成片总长与时间线不符')


def lively_headers(item, font_path, top_k):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    color = ACCENTS[(item['rank']-1) % len(ACCENTS)]
    title = lively_title(item)
    font = ImageFont.truetype(str(font_path), 35)
    small = ImageFont.truetype(str(font_path), 18)
    badge = ImageFont.truetype(str(font_path), 42)
    frames = []
    for i in range(16):
        p = min(1., i/12)
        canvas = Image.new('RGB', (720, 110), INK)
        draw = ImageDraw.Draw(canvas)
        draw.polygon([(570,0),(602,0),(550,110),(518,110)], fill=(23,41,57))
        draw.polygon([(655,0),(674,0),(622,110),(603,110)], fill=(23,41,57))
        x = round(20-130*(1-p)**3)
        draw.rounded_rectangle((x+4,19,x+102,99), radius=19, fill=(4,9,20))
        draw.rounded_rectangle((x,14,x+98,94), radius=19, fill=color)
        draw.text((x+11,19), f"#{item['rank']}", font=badge, fill=INK)
        tx = round(136+26*(1-p)**3)
        draw.text((tx+2,17), title, font=font, fill=(2,5,12))
        draw.text((tx,15), title, font=font, fill=color)
        draw.text((137,67), f'VOLLEYMOLE  /  TOP {top_k}  /  好球不断', font=small, fill=CREAM)
        draw.rounded_rectangle((136,100,136+round(120*p),104), radius=2, fill=color)
        frames.append(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    return frames


def overlay_asset(path, kind, font_path, index=0):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGBA', (720,1280), (0,0,0,0))
    draw = ImageDraw.Draw(canvas)
    color = ACCENTS[index % len(ACCENTS)]
    if kind == 'teaser':
        draw.rectangle((0,0,720,109), fill=INK+(255,))
        draw.rounded_rectangle((20,20,130,90), radius=17, fill=color+(255,))
        draw.text((32,30), '高能', font=ImageFont.truetype(str(font_path),32), fill=INK)
        draw.text((151,12), '先看这几下！', font=ImageFont.truetype(str(font_path),40), fill=color)
        draw.text((155,67), '精彩抢先看  /  马上进入倒计时', font=ImageFont.truetype(str(font_path),21), fill=CREAM)
        draw.rounded_rectangle((186,1180,534,1242),radius=25,fill=INK+(235,))
        draw.text((220,1189), '别眨眼，好球来了！',font=ImageFont.truetype(str(font_path),27),fill=CREAM)
    else:
        draw.rounded_rectangle((21,126,411,191),radius=18,fill=INK+(235,))
        draw.text((40,136),'再看一次  /  0.67×',font=ImageFont.truetype(str(font_path),30),fill=(255,198,70))
        draw.rectangle((0,111,719,874), outline=(255,198,70,255),width=4)
    canvas.save(path)


def effect_clip(source, path, segment, relative_start, overlay):
    rate = segment['playback_rate']; duration = segment['duration_sec']
    span = segment['source_end_sec']-segment['source_start_sec']
    graph = (f'[0:v]setpts=(PTS-STARTPTS)/{rate},fps=30,tpad=stop_mode=clone:stop_duration=1,'
             f'trim=end_frame={segment["output_frames"]},setsar=1[v];'
             '[v][1:v]overlay=0:0:shortest=1,format=yuv420p[outv];'
             f'[0:a]asetpts=PTS-STARTPTS,atempo={rate},apad,atrim=duration={duration},'
             f'afade=t=in:d=0.025,afade=t=out:st={max(0,duration-.04)}:d=0.04[outa]')
    subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-ss',str(relative_start),'-t',str(span),'-i',source,
        '-loop','1','-framerate','30','-i',str(overlay),'-filter_complex_threads','2','-filter_complex',graph,
        '-map','[outv]','-map','[outa]','-t',str(duration),'-r','30','-c:v','libx264','-preset','fast','-crf','20',
        '-threads','4','-c:a','aac','-ar','48000','-ac','2','-b:a','192k','-movflags','+faststart',str(path)],check=True)


def transition_clip(previous, following, path, segment, font_path):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from media_worker import frame_at
    before = frame_at(previous['path'],max(0,previous['duration_sec']-1/FPS),720)
    after = frame_at(following['path'],0,720)
    ys,xs = np.ogrid[:1280,:720]; diagonal = xs+.25*ys
    color = ACCENTS[(segment['next_rank']-1) % len(ACCENTS)]
    label = Image.new('RGB',(390,90),INK)
    draw = ImageDraw.Draw(label)
    draw.text((23,13),f'下一球  #{segment["next_rank"]}',font=ImageFont.truetype(str(font_path),43),fill=color)
    label = cv2.cvtColor(np.asarray(label),cv2.COLOR_RGB2BGR)
    duration = segment['duration_sec']
    # A quiet synthesized whoosh; main-rally and replay sound remain source audio.
    sound = f'anoisesrc=d={duration}:c=pink:r=48000:a=0.10:seed=42,highpass=f=800,lowpass=f=4500,afade=t=in:d=0.10,afade=t=out:st=0.15:d=0.25'
    process = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','720x1280','-r','30','-i','pipe:0',
        '-f','lavfi','-i',sound,'-t',str(duration),'-c:v','libx264','-preset','fast','-crf','20','-threads','4',
        '-pix_fmt','yuv420p','-c:a','aac','-ar','48000','-ac','2','-b:a','192k',str(path)],stdin=subprocess.PIPE)
    try:
        for i in range(segment['output_frames']):
            p = i/(segment['output_frames']-1)
            edge = -140+1320*p
            frame = np.where((diagonal<edge)[...,None],after,before).copy()
            if 0<i<segment['output_frames']-1:
                frame[abs(diagonal-edge)<58] = color[::-1]
                frame[(diagonal-edge>58)&(diagonal-edge<82)] = CREAM[::-1]
                x = round(-390+1500*p)
                left,right=max(0,x),min(720,x+390)
                if right>left: frame[560:650,left:right] = label[:,left-x:right-x]
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait(): raise RuntimeError('连接动画编码失败')
    finally:
        if process.poll() is None: process.terminate();process.wait()


def render_lively(directory, font):
    from media_worker import render_clip, concatenate_segments
    from schemas import validate_decision
    started = time.perf_counter()
    manifest = read_json(directory/'match_manifest.json'); decision = read_json(directory/'edit_decision.json')
    validate_decision(decision,manifest,directory,len(decision['selected']))
    segments = build_timeline(decision,manifest)
    extras = directory/'extras_lively';extras.mkdir(exist_ok=True)
    clips=[]
    for item in reversed(decision['selected']):
        print(f"活力版 #{item['rank']} {item['rally_id']}",flush=True)
        try: clip=render_clip(directory,item,manifest,font,style='lively')
        except (ValueError,IndexError) as exc:
            clip=render_clip(directory,item,manifest,font,center_only=True,style='lively')
            clip['fallback_error']=type(exc).__name__
        clips.append(clip)
    base_elapsed = time.perf_counter()-started
    by_rank={c['rank']:c for c in clips}
    for segment in segments:
        kind=segment['kind']
        if kind=='rally':
            segment['path']=by_rank[segment['rank']]['path']
            by_rank[segment['rank']]['timeline_start_sec']=segment['timeline_start_sec']
        elif kind in ('teaser','replay'):
            print(f"{kind} #{segment['rank']}",flush=True)
            overlay=extras/f'{kind}_{segment["index"]:02d}.png'
            overlay_asset(overlay,kind,font,segment['index'])
            path=extras/f'{kind}_{segment["index"]:02d}.mp4'
            base=by_rank[segment['rank']]
            effect_clip(base['path'],path,segment,segment['source_start_sec']-base['source_start_sec'],overlay)
            segment['path']=str(path)
        else:
            path=extras/f'transition_{segment["index"]:02d}.mp4'
            previous=segments[segment['index']-1]
            transition_clip(previous,by_rank[segment['next_rank']],path,segment,font)
            segment['path']=str(path)
    effects_elapsed=time.perf_counter()-started-base_elapsed
    output=directory/f'top{len(clips)}_lively.mp4'
    concatenate_segments(segments,output)
    render_elapsed=time.perf_counter()-started
    save_json(directory/'render_report_lively.json',{'output':str(output),'style':'lively','order':'countdown',
        'clips':clips,'segments':segments,'expected_duration_sec':sum(s['output_frames'] for s in segments)/FPS,
        'teaser_duration_sec':sum(s['duration_sec'] for s in segments if s['kind']=='teaser'),
        'replay_speed':REPLAY_SPEED,'render_timing_sec':{'rally_render':round(base_elapsed,3),
        'teaser_replay_transition':round(effects_elapsed,3),'assembly':round(render_elapsed-base_elapsed-effects_elapsed,3),
        'total':round(render_elapsed,3)},'audio':'Source sound; time-stretched replay audio; quiet synthesized transition whoosh.'})
