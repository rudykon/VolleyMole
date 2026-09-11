"""PTS-based video sampling and rendering in the unified VolleyMole environment."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import ROOT, DEFAULT_FONT, read_json, save_json
from .art_themes import THEME_IDS, get_theme
from .title_templates import TEMPLATE_IDS, get_template
from .transitions import STYLE_IDS, validate_style
from .design_suites import SUITE_IDS, resolve_design, palette
from .sources import source_for

FONT = str(DEFAULT_FONT)


def frame_at(video, seconds, width=640,lossless=False):
    data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(seconds),'-i',str(video),'-frames:v','1',
                                  '-vf',f'scale={width}:-2','-f','image2pipe','-vcodec','png' if lossless else 'mjpeg','pipe:1'])
    pixels=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if pixels is None: raise ValueError(f'关键帧无法解码：{seconds}')
    return pixels


def previews(directory, rally_ids=None, workers=1):
    from concurrent.futures import ThreadPoolExecutor
    manifest=read_json(directory/'match_manifest.json')
    outputs=[]
    jobs=[]
    for r in manifest['rallies']:
        if rally_ids is not None and r['rally_id'] not in rally_ids:
            continue
        for when,name in zip(r['preview_times_sec'],r['preview_frames']):
            jobs.append((source_for(manifest,r['rally_id'])['path'],when,name))
            outputs.append(name)
    def extract(job):
        video,when,name=job
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),frame_at(video,when)):
            raise RuntimeError(f'无法保存关键帧：{path}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(extract,jobs))
    save_json(directory/'previews/index.json',{'files':outputs})


def render_clip(directory, item, manifest, font_path, center_only=False, style='classic', art_theme='default',title_template='legacy',design_suite='custom',quality='720p'):
    from .quality import get_quality
    q=get_quality(quality)
    scale=q.width/720
    scaled=lambda value: round(value*scale)
    from .camera import smooth_values, crop_frame
    r=next(r for r in manifest['rallies'] if r['rally_id']==item['rally_id'])
    track=read_json(directory/r['tracking_json'])
    source=source_for(manifest,item['rally_id']);w,h=source['width'],source['height']
    if not Path(source['path']).is_file():
        raise ValueError(f'找不到原片：{source["path"]}')
    start,end=item['clip_start_sec'],item['clip_end_sec']
    frames=math.ceil((end-start)*30-1e-6)
    # All output streams use the same source-relative PTS at 30 Hz.
    times=start+np.arange(frames)/30
    samples=np.array([s[:3] for s in track['samples'] if s[3]],dtype=float)
    mode='ball_follow'
    if center_only or samples.size==0:
        centers=np.full(frames,w/2);mode='center_fallback'
    else:
        centers=np.interp(times,samples[:,0],samples[:,1])
        centers=smooth_values(centers,'moving_avg',15,2)[:frames]
        # Long occlusions settle gently towards center instead of chasing noise.
        nearest=np.searchsorted(samples[:,0],times).clip(0,len(samples)-1)
        prev=(nearest-1).clip(0,len(samples)-1)
        gaps=np.minimum(abs(samples[nearest,0]-times),abs(samples[prev,0]-times))
        blend=np.clip((gaps-.8)/1.5,0,1)
        centers=centers*(1-blend)+(w/2)*blend
        # Limit camera acceleration through a second low-pass pass.
        centers=smooth_values(centers,'moving_avg',9,2)[:frames]
    out=directory/('clips_lively' if style=='lively' else 'clips')/f"rank_{item['rank']:02d}.mp4"
    out.parent.mkdir(exist_ok=True)
    temp=out.with_name(out.stem+'.tmp.mp4')
    log=out.with_suffix('.log')
    header_h,detail_h,overview_h=110,765,405
    detail_top=0 if style=='lively' else header_h
    picture_h=header_h+detail_h-detail_top
    # The inset keeps all players visible even when the detail crop follows a ball.
    font=ImageFont.truetype(str(font_path),32)
    small=ImageFont.truetype(str(font_path),20)
    header=Image.new('RGB',(720,header_h),(15,23,36))
    draw=ImageDraw.Draw(header)
    draw.rounded_rectangle((20,18,114,92),radius=12,fill=(248,190,54))
    draw.text((35,27),f"#{item['rank']}",font=ImageFont.truetype(str(font_path),42),fill=(18,23,31))
    title=item['title']
    while draw.textlength(title,font=font)>560:
        font=ImageFont.truetype(str(font_path),font.size-1)
    draw.text((134,18),title,font=font,fill='white')
    decision = read_json(directory/'edit_decision.json')
    count = len(decision['selected'])
    caption = '排球趣味时刻' if decision.get('collection') == 'bloopers' else f'日常排球{count}佳球'
    draw.text((135,65),'VOLLEYMOLE  /  '+caption,font=small,fill=(169,187,207))
    header_array=cv2.cvtColor(np.asarray(header),cv2.COLOR_RGB2BGR)
    lively=None
    if style=='lively':
        from .presentation import lively_headers
        from .illustrated import composite_header
        lively=lively_headers(item,font_path,count,art_theme,title_template,design_suite,decision.get('collection','highlights'))
    accent=palette(art_theme,design_suite).colors[1][::-1] if style=='lively' else (54,190,248)
    court_label=None
    if style=='lively' and design_suite!='custom':
        from .title_templates import text_layer
        theme=palette(art_theme,design_suite)
        label=text_layer('全场视角' if get_template(title_template).language=='zh' else 'FULL COURT',16,theme.cream,260,title_template,outline=theme.ink)
        court_label=cv2.cvtColor(np.asarray(label),cv2.COLOR_RGBA2BGRA)
    # Use a top-anchored court detail to discard empty foreground floor. The full
    # original frame is always available in the inset immediately below it.
    detail_source_h=min(h,max(int(.68*h),int(np.percentile(samples[:,2],95)+.3*h) if samples.size else h))
    detail_source_h=min(h,max(detail_source_h,round(h*.72)))
    crop_width=min(w,round(detail_source_h*720/picture_h))
    decoder_cmd=['ffmpeg','-v','error','-threads','2','-ss',str(start),'-i',source['path'],'-t',str(frames/30),
                 '-vf','fps=30','-frames:v',str(frames),'-f','rawvideo','-pix_fmt','bgr24','pipe:1']
    encoder_cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{q.width}x{q.height}','-r','30','-i','pipe:0']
    if source['has_audio']:
        encoder_cmd+=['-ss',str(start),'-i',source['path'],'-map','0:v:0','-map','1:a:0',
                      '-af',f'aresample=48000:async=1:first_pts=0,apad,atrim=duration={frames/30}',
                      '-c:a','aac','-b:a','192k','-ac','2']
    else:
        encoder_cmd+=['-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k']
    encoder_cmd+=['-t',str(frames/30),'-c:v','libx264','-preset','fast','-crf',str(q.crf),'-threads','4',
                  '-pix_fmt','yuv420p','-vf','setsar=1','-movflags','+faststart',str(temp)]
    done=0
    with log.open('w') as errors:
        decoder=subprocess.Popen(decoder_cmd,stdout=subprocess.PIPE,stderr=errors)
        encoder=subprocess.Popen(encoder_cmd,stdin=subprocess.PIPE,stderr=errors)
        try:
            for i in range(frames):
                data=decoder.stdout.read(w*h*3)
                if len(data)!=w*h*3: raise ValueError(f'片段视频提前结束：{i}/{frames}')
                frame=np.frombuffer(data,np.uint8).reshape(h,w,3)
                canvas=np.empty((q.height,q.width,3),np.uint8)
                crop=crop_frame(frame[:detail_source_h],int(centers[i]),crop_width,'none')
                canvas[scaled(detail_top):scaled(875)]=cv2.resize(crop,(q.width,scaled(875)-scaled(detail_top)),interpolation=cv2.INTER_LANCZOS4 if q.width>crop.shape[1] else cv2.INTER_AREA)
                if lively:
                    canvas[:scaled(header_h)]=composite_header(canvas[:scaled(header_h)],cv2.resize(lively[min(i,len(lively)-1)],(q.width,scaled(header_h)),interpolation=cv2.INTER_LANCZOS4))
                else:
                    canvas[:scaled(header_h)]=cv2.resize(header_array,(q.width,scaled(header_h)),interpolation=cv2.INTER_LANCZOS4)
                canvas[scaled(875):]=cv2.resize(frame,(q.width,q.height-scaled(875)),interpolation=cv2.INTER_AREA)
                cv2.rectangle(canvas,(0,scaled(875)),(q.width,scaled(878)),accent,-1)
                if court_label is not None:
                    label=cv2.resize(court_label,(scaled(court_label.shape[1]),scaled(court_label.shape[0])),interpolation=cv2.INTER_LANCZOS4)
                    lh,lw=label.shape[:2];y,x=scaled(881),scaled(12)
                    canvas[y:y+lh,x:x+lw]=composite_header(canvas[y:y+lh,x:x+lw],label)
                else:
                    cv2.putText(canvas,'FULL COURT',(scaled(16),scaled(904)),cv2.FONT_HERSHEY_SIMPLEX,.55*scale,(255,255,255),max(1,scaled(1)),cv2.LINE_AA)
                cv2.rectangle(canvas,(0,scaled(1275)),(round(q.width*(i+1)/frames),q.height-1),accent,-1)
                encoder.stdin.write(canvas.tobytes());done+=1
            encoder.stdin.close()
            if decoder.wait()!=0 or encoder.wait()!=0: raise RuntimeError(f'渲染失败，见 {log}')
        finally:
            for process in (decoder,encoder):
                if process.poll() is None: process.terminate();process.wait()
    temp.replace(out)
    report={'output_quality':q.report(),'rally_id':r['rally_id'],'rank':item['rank'],'path':str(out),'source_start_sec':start,
            'source_end_sec':end,'output_frames':done,'duration_sec':done/30,'crop_mode':mode,
            'silent_source':not source['has_audio'],'source_time_per_frame':'source_start_sec + frame / 30',
            'follow_center_min':float(min(centers)), 'follow_center_max':float(max(centers))}
    if style=='lively':
        report['art_theme']=art_theme
        report['title_template']=title_template
        report['design_suite']=design_suite
        report['header_background']='transparent'
        report['detail_layout']={'top':detail_top,'height':picture_h,'overview_top':header_h+detail_h,
                                 'source_height':detail_source_h,'source_crop_width':crop_width}
        # Independent verification can reconstruct the live picture behind the UI.
        report['header_source_samples']=[{'output_frame':i,'source_sec':start+i/30,
            'crop_left':max(0,min(int(centers[i])-crop_width//2,w-crop_width))}
            for i in sorted({min(15,frames-1),frames//2,max(0,frames-15)})]
    if 'sources' in manifest:
        report.update(source_id=r['source_id'],source_set=r['source_set'],source_path=source['path'],time_basis='source-local seconds')
    save_json(out.with_suffix('.json'),report)
    return report


def render(directory,font,style='classic',workers=1,art_theme=None,title_template=None,transition_style=None,design_suite=None,design_language=None,quality=None):
    from concurrent.futures import ThreadPoolExecutor
    from .quality import get_quality,DEFAULT_QUALITY
    config=read_json(directory/'run_config.json') if (directory/'run_config.json').is_file() else {}
    quality=quality or config.get('quality',DEFAULT_QUALITY)
    q=get_quality(quality)
    if style=='lively':
        from .presentation import render_lively
        config=read_json(directory/'run_config.json') if (directory/'run_config.json').is_file() else {}
        if design_suite is None:design_suite=config.get('design_suite','custom')
        if design_language is None:design_language=config.get('design_language','zh')
        if art_theme is None:
            art_theme=config.get('art_theme','default')
        if title_template is None:title_template=config.get('title_template','legacy')
        if transition_style is None:transition_style=config.get('transition_style','fade')
        art_theme,title_template,transition_style=resolve_design(design_suite,design_language,art_theme,title_template,transition_style)
        validate_style(transition_style)
        get_template(title_template)
        return render_lively(directory,font,workers,art_theme,title_template=title_template,transition_style=transition_style,design_suite=design_suite,quality=quality)
    if art_theme not in (None,'default') or title_template not in (None,'legacy') or transition_style not in (None,'fade') or design_suite not in (None,'custom'):
        raise ValueError('--art-theme / --title-template / --transition-style / --design-suite 仅用于 lively 呈现')
    manifest=read_json(directory/'match_manifest.json');decision=read_json(directory/'edit_decision.json')
    from .schemas import validate_decision
    validate_decision(decision,manifest,directory,len(decision['selected']))
    def render_item(item):
        print(f"渲染 #{item['rank']} {item['rally_id']}",flush=True)
        try:
            result=render_clip(directory,item,manifest,font,quality=quality)
        except (ValueError,IndexError) as exc:
            result=render_clip(directory,item,manifest,font,center_only=True,quality=quality)
            result['fallback_error']=type(exc).__name__
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        clips=list(pool.map(render_item,reversed(decision['selected'])))
    output=directory/f'top{len(clips)}.mp4'
    concatenate_segments(clips,output,quality)
    save_json(directory/'render_report.json',{'output_quality':q.report(),'output':str(output),'order':'countdown','clips':clips,'expected_duration_sec':sum(r['duration_sec'] for r in clips)})


def concatenate_segments(clips,output,quality='720p'):
    from .quality import get_quality
    q=get_quality(quality)
    # Decode and concatenate exact video/audio durations. Concatenating MP4
    # packets uses AAC-rounded container lengths and accumulates audio drift.
    temp=output.with_name(output.stem+'.tmp.mp4')
    command=['ffmpeg','-y','-v','error']
    filters=[]
    for i,clip in enumerate(clips):
        command+=['-threads','2','-i',clip['path']]
        filters += [f'[{i}:v]setpts=PTS-STARTPTS[v{i}]',
                    f'[{i}:a]atrim=duration={clip["duration_sec"]},asetpts=PTS-STARTPTS[a{i}]']
    filters.append(''.join(f'[v{i}][a{i}]' for i in range(len(clips)))+f'concat=n={len(clips)}:v=1:a=1[v][a]')
    command+=['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
              '-c:v','libx264','-preset','fast','-crf',str(q.crf),'-threads','4','-pix_fmt','yuv420p','-r','30',
              '-c:a','aac','-b:a','192k','-movflags','+faststart',str(temp)]
    subprocess.run(command,check=True)
    temp.replace(output)


if __name__=='__main__':
    from .quality import QUALITY_IDS
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['previews','render'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--quality',choices=QUALITY_IDS,help='成片画质；省略时读取运行配置，默认 1080p')
    parser.add_argument('--font',default=FONT)
    parser.add_argument('--style',choices=['classic','lively'],default='classic')
    parser.add_argument('--art-theme',choices=THEME_IDS,help='lively 插画套装；省略时读取运行配置，旧运行使用 default')
    parser.add_argument('--title-template',choices=TEMPLATE_IDS,help='中英标题模板；省略时读取运行配置，旧运行使用 legacy')
    parser.add_argument('--transition-style',choices=STYLE_IDS,help='动画转场；省略时读取运行配置，旧运行使用 fade')
    parser.add_argument('--design-suite',choices=SUITE_IDS,help='完整设计套装；省略时读取运行配置，优先于三个单项选项')
    parser.add_argument('--design-language',choices=('zh','en'),help='套装语言；省略时读取运行配置，默认 zh')
    parser.add_argument('--rally-ids',nargs='*')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=1)
    args=parser.parse_args();directory=args.run.resolve()
    cv2.setNumThreads(2)
    if args.kind=='previews':previews(directory,args.rally_ids,args.workers)
    else:render(directory,args.font,args.style,args.workers,args.art_theme,args.title_template,args.transition_style,args.design_suite,args.design_language,args.quality)
