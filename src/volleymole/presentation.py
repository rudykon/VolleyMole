"""A source-traceable lively edit: teaser, full rallies, slow replays and wipes."""
import math
from pathlib import Path
import subprocess
import time
from .common import read_json, save_json, identity
from .illustrated import TRANSITION_FRAMES, TRANSITION_RAMP_FRAMES, asset_paths, prepare_brand, rank_label, illustration_for

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
    """Use the candidate keyframe; its peak is a hypothesis, not highlight truth."""
    start, end = item['clip_start_sec'], item['clip_end_sec']
    peak = rally['preview_times_sec'][1]
    if not math.isfinite(peak) or not start <= peak <= end:
        peak = (start + end) / 2
    length = min(length, end-start)
    a = min(max(start, peak-length*.6), end-length)
    return {'source_start_sec': a, 'source_end_sec': a+length,
            'peak_sec': peak, 'peak_evidence': 'candidate_preview_peak',
            'peak_selection_uncertainty':'Action-detector timestamp or candidate midpoint; not guaranteed the most exciting moment.',
            'rally_id': item['rally_id'], 'rank': item['rank']}


def build_timeline(decision, manifest):
    by_id = {r['rally_id']: r for r in manifest['rallies']}
    segments = []
    def add(kind, frames, **data):
        data['rank_label']=rank_label(data.get('rank',data.get('next_rank')),len(decision['selected']))
        segments.append(dict(kind=kind, output_frames=frames, duration_sec=frames/FPS, **data))
    # Five one-second highlights, followed by readable full-screen title cards.
    for item in reversed(decision['selected'][:5]):
        window = peak_window(item, by_id[item['rally_id']], 1.)
        window['visual_review_used']=decision.get('ranking_mode') in ('multimodal_api','vision_then_text_api')
        add('teaser', 30, playback_rate=1., **window)
    for item in reversed(decision['selected']):
        add('transition', TRANSITION_FRAMES, next_rank=item['rank'],top_k=len(decision['selected']),
            original_title=item['title'],display_title=lively_title(item),
            illustration=illustration_for(item).name,
            title_hold_sec=(TRANSITION_FRAMES-2*TRANSITION_RAMP_FRAMES)/FPS)
        frames = math.ceil((item['clip_end_sec']-item['clip_start_sec'])*FPS-1e-6)
        add('rally', frames, rank=item['rank'], rally_id=item['rally_id'], playback_rate=1.,
            source_start_sec=item['clip_start_sec'], source_end_sec=item['clip_end_sec'])
        window = peak_window(item, by_id[item['rally_id']])
        window['visual_review_used']=decision.get('ranking_mode') in ('multimodal_api','vision_then_text_api')
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
    if sum(s['output_frames'] for s in segments if s['kind']=='teaser')!=150:
        raise ValueError('快切片头必须为5秒')
    transitions=[s for s in segments if s['kind']=='transition']
    if [s['next_rank'] for s in transitions]!=list(reversed(expected)):
        raise ValueError('每个完整回合前必须有可读的标题转场')
    if any(s['output_frames']!=TRANSITION_FRAMES or s['title_hold_sec']<2 for s in transitions):
        raise ValueError('转场必须为3秒，标题完整展示不少于2秒')
    for segment in segments:
        number=segment.get('rank',segment.get('next_rank'))
        if segment.get('rank_label')!=rank_label(number,len(expected)):
            raise ValueError('回合或转场名次标识与实际排名不一致')
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
    from .illustrated import headers
    return headers(item,font_path,top_k,lively_title(item))


def overlay_asset(path, kind, font_path, index=0):
    from .illustrated import overlay
    return overlay(path,kind,font_path,index)


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
    from .illustrated import encode_transition
    return encode_transition(previous,following,path,segment,font_path)


def render_lively(directory, font):
    from .media_worker import render_clip, concatenate_segments
    from .schemas import validate_decision
    started = time.perf_counter()
    prepare_brand()
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
        clip['display_title']=lively_title(item)
        clip['rank_label']=rank_label(item['rank'],len(decision['selected']))
        clip['illustration']=illustration_for(item).name
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
    save_json(directory/'render_report_lively.json',{'output':str(output),'style':'lively','design_revision':5,'order':'countdown',
        'font_path':str(font),
        'header_background':'transparent',
        'replay_caption_background':'transparent',
        'illustration_assets':[identity(p) for p in asset_paths()],
        'clips':clips,'segments':segments,'expected_duration_sec':sum(s['output_frames'] for s in segments)/FPS,
        'teaser_duration_sec':sum(s['duration_sec'] for s in segments if s['kind']=='teaser'),
        'replay_speed':REPLAY_SPEED,'render_timing_sec':{'rally_render':round(base_elapsed,3),
        'teaser_replay_transition':round(effects_elapsed,3),'assembly':round(render_elapsed-base_elapsed-effects_elapsed,3),
        'total':round(render_elapsed,3)},'audio':'Source sound; time-stretched replay audio; quiet synthesized transition whoosh.'})
