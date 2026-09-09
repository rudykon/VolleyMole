"""PTS-based video sampling and rendering in the tracking project's environment."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import ROOT, read_json, save_json

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


def frame_at(video, seconds, width=640):
    data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(seconds),'-i',str(video),'-frames:v','1',
                                  '-vf',f'scale={width}:-2','-f','image2pipe','-vcodec','mjpeg','pipe:1'])
    pixels=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if pixels is None: raise ValueError(f'关键帧无法解码：{seconds}')
    return pixels


def previews(directory):
    manifest=read_json(directory/'match_manifest.json')
    video=manifest['source']['path']; outputs=[]
    for r in manifest['rallies']:
        for when,name in zip(r['preview_times_sec'],r['preview_frames']):
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path),frame_at(video,when)):
                raise RuntimeError(f'无法保存关键帧：{path}')
            outputs.append(name)
    save_json(directory/'previews/index.json',{'files':outputs})


def render_clip(directory, item, manifest, font_path, center_only=False):
    sys.path.insert(0,str(ROOT/'tools/fast-volleyball-tracking-inference/src'))
    from make_reels import smooth_values, crop_frame
    r=next(r for r in manifest['rallies'] if r['rally_id']==item['rally_id'])
    track=read_json(directory/r['tracking_json'])
    source=manifest['source'];w,h=source['width'],source['height']
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
    out=directory/'clips'/f"rank_{item['rank']:02d}.mp4"
    out.parent.mkdir(exist_ok=True)
    temp=out.with_name(out.stem+'.tmp.mp4')
    log=directory/'clips'/f"rank_{item['rank']:02d}.log"
    header_h,detail_h,overview_h=110,765,405
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
    draw.text((135,65),'VOLLEYMOLE  /  日常排球五佳球' if len(read_json(directory/'edit_decision.json')['selected'])==5 else 'VOLLEYMOLE  /  日常排球十佳球',font=small,fill=(169,187,207))
    header_array=cv2.cvtColor(np.asarray(header),cv2.COLOR_RGB2BGR)
    # Use a top-anchored court detail to discard empty foreground floor. The full
    # original frame is always available in the inset immediately below it.
    detail_source_h=min(h,max(int(.68*h),int(np.percentile(samples[:,2],95)+.3*h) if samples.size else h))
    detail_source_h=min(h,max(detail_source_h,round(h*.72)))
    crop_width=min(w,round(detail_source_h*720/detail_h))
    decoder_cmd=['ffmpeg','-v','error','-threads','2','-ss',str(start),'-i',source['path'],'-t',str(frames/30),
                 '-vf','fps=30','-frames:v',str(frames),'-f','rawvideo','-pix_fmt','bgr24','pipe:1']
    encoder_cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','720x1280','-r','30','-i','pipe:0']
    if source['has_audio']:
        encoder_cmd+=['-ss',str(start),'-i',source['path'],'-map','0:v:0','-map','1:a:0',
                      '-af',f'aresample=48000:async=1:first_pts=0,apad,atrim=duration={frames/30}',
                      '-c:a','aac','-b:a','192k','-ac','2']
    else:
        encoder_cmd+=['-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k']
    encoder_cmd+=['-t',str(frames/30),'-c:v','libx264','-preset','fast','-crf','20','-threads','4',
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
                canvas=np.empty((1280,720,3),np.uint8);canvas[:header_h]=header_array
                crop=crop_frame(frame[:detail_source_h],int(centers[i]),crop_width,'none')
                canvas[header_h:header_h+detail_h]=cv2.resize(crop,(720,detail_h),interpolation=cv2.INTER_AREA)
                canvas[header_h+detail_h:]=cv2.resize(frame,(720,overview_h),interpolation=cv2.INTER_AREA)
                cv2.rectangle(canvas,(0,header_h+detail_h),(720,header_h+detail_h+3),(54,190,248),-1)
                cv2.putText(canvas,'FULL COURT',(16,header_h+detail_h+29),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                cv2.rectangle(canvas,(0,1275),(round(720*(i+1)/frames),1279),(54,190,248),-1)
                encoder.stdin.write(canvas.tobytes());done+=1
            encoder.stdin.close()
            if decoder.wait()!=0 or encoder.wait()!=0: raise RuntimeError(f'渲染失败，见 {log}')
        finally:
            for process in (decoder,encoder):
                if process.poll() is None: process.terminate();process.wait()
    temp.replace(out)
    report={'rally_id':r['rally_id'],'rank':item['rank'],'path':str(out),'source_start_sec':start,
            'source_end_sec':end,'output_frames':done,'duration_sec':done/30,'crop_mode':mode,
            'silent_source':not source['has_audio'],'source_time_per_frame':'source_start_sec + frame / 30',
            'follow_center_min':float(min(centers)), 'follow_center_max':float(max(centers))}
    save_json(out.with_suffix('.json'),report)
    return report


def render(directory,font):
    manifest=read_json(directory/'match_manifest.json');decision=read_json(directory/'edit_decision.json')
    from schemas import validate_decision
    validate_decision(decision,manifest,directory,len(decision['selected']))
    clips=[]
    for item in reversed(decision['selected']):
        print(f"渲染 #{item['rank']} {item['rally_id']}",flush=True)
        try:
            result=render_clip(directory,item,manifest,font)
        except (ValueError,IndexError) as exc:
            result=render_clip(directory,item,manifest,font,center_only=True)
            result['fallback_error']=type(exc).__name__
        clips.append(result)
    # Decode and concatenate exact video/audio durations. Concatenating MP4
    # packets uses AAC-rounded container lengths and accumulates audio drift.
    output=directory/f'top{len(clips)}.mp4';temp=output.with_name(output.stem+'.tmp.mp4')
    command=['ffmpeg','-y','-v','error']
    filters=[]
    for i,clip in enumerate(clips):
        command+=['-threads','2','-i',clip['path']]
        filters += [f'[{i}:v]setpts=PTS-STARTPTS[v{i}]',
                    f'[{i}:a]atrim=duration={clip["duration_sec"]},asetpts=PTS-STARTPTS[a{i}]']
    filters.append(''.join(f'[v{i}][a{i}]' for i in range(len(clips)))+f'concat=n={len(clips)}:v=1:a=1[v][a]')
    command+=['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
              '-c:v','libx264','-preset','fast','-crf','19','-threads','4','-pix_fmt','yuv420p','-r','30',
              '-c:a','aac','-b:a','192k','-movflags','+faststart',str(temp)]
    subprocess.run(command,check=True)
    temp.replace(output)
    save_json(directory/'render_report.json',{'output':str(output),'order':'countdown','clips':clips,'expected_duration_sec':sum(r['duration_sec'] for r in clips)})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['previews','render'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--font',default=FONT)
    args=parser.parse_args();directory=args.run.resolve()
    cv2.setNumThreads(2)
    if args.kind=='previews':previews(directory)
    else:render(directory,args.font)
