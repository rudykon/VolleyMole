"""Independent content checks against the source, beyond stream duration checks."""
import argparse
from pathlib import Path
import subprocess
import cv2
import numpy as np
from scipy.signal import correlate, correlation_lags
from .common import read_json, save_json
from .media_worker import frame_at
from .sources import source_for


def audio_at(path,start,duration=2.):
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(path),'-t',str(duration),
                                 '-vn','-ac','1','-ar','16000','-f','f32le','pipe:1'])
    return np.frombuffer(raw,np.float32).astype(float)


def audio_alignment(original,rendered):
    n=min(len(original),len(rendered));x=original[:n]-original[:n].mean();y=rendered[:n]-rendered[:n].mean()
    if np.linalg.norm(x)<1e-6 or np.linalg.norm(y)<1e-6:
        return {'status':'silent_window','lag_ms':0,'correlation':None}
    cc=correlate(y,x,method='fft');lags=correlation_lags(n,n)
    allowed=abs(lags)<=1600
    idx=np.flatnonzero(allowed)[np.argmax(cc[allowed])]
    return {'status':'checked','lag_ms':float(lags[idx]/16),'correlation':float(cc[idx]/(np.linalg.norm(x)*np.linalg.norm(y)))}


def single_view_error(source,clip,when,pixels,kind='rally'):
    """Reconstruct a recorded source crop, including the bottom of the picture.

    Compare within one 30 Hz frame to allow the original VFR stream and the
    slow-motion fps filter to round their sampling instants differently.
    No rendered clip is used as the expected source picture.
    """
    layout=clip['detail_layout'];lefts=clip['camera_crop_left']
    top,height=layout['top'],layout['height']
    width,source_height=layout['source_crop_width'],layout['source_height']
    if (top,height,layout.get('overview_top')) not in ((0,1280,None),(110,1170,None)):
        raise ValueError('单画面布局不完整或仍含全场小窗')
    if len(lefts)!=clip['output_frames'] or not 0<width<=source['width'] or not 0<source_height<=source['height']:
        raise ValueError('缺少有效的逐帧构图记录')
    index=round((when-clip['source_start_sec'])*30)
    errors=[]
    for frame in sorted({max(0,min(len(lefts)-1,index+delta)) for delta in (-1,0,1)}):
        left=lefts[frame]
        if type(left) is not int or not 0<=left<=source['width']-width:
            raise ValueError('构图记录越出源画面')
        original=frame_at(source['path'],clip['source_start_sec']+frame/30,source['width'],lossless=True)
        target=np.zeros((1280,720,3),np.uint8)
        target[top:]=cv2.resize(original[:source_height,left:left+width],(720,height),interpolation=cv2.INTER_AREA)
        if kind=='teaser':
            # The teaser removes the inherited rank header and fills the frame
            # with one aspect-preserving crop before adding its own lettering.
            zoom_width=round(720*1280/1170/2)*2
            enlarged=cv2.resize(target[110:],(zoom_width,1280),interpolation=cv2.INTER_CUBIC)
            x=(zoom_width-720)//2;target=enlarged[:,x:x+720]
        bottom=1165 if kind=='teaser' else 1270
        region=(slice(210,bottom),slice(12,708))
        error=float(np.mean(abs(pixels[region].astype(float)-target[region].astype(float))))
        errors.append((error,frame))
    error,frame=min(errors)
    return {'picture_mean_abs_error':error,'matched_source_sec':clip['source_start_sec']+frame/30}


def check(directory,style='classic'):
    suffix='_lively' if style=='lively' else ''
    manifest=read_json(directory/'match_manifest.json');report=read_json(directory/f'render_report{suffix}.json')
    results=[];offset=0.
    by_rank={clip['rank']:clip for clip in report['clips']}
    for clip in report['clips']:
        source=source_for(manifest,clip['rally_id'])
        offset=clip.get('timeline_start_sec',offset)
        samples=[]
        for relative in (.5,clip['duration_sec']/2,clip['duration_sec']-1.):
            output=frame_at(clip['path'],relative,720)
            when=clip['source_start_sec']+relative
            if clip.get('view_layout')=='single':
                result=single_view_error(source,clip,when,output)
                combined=frame_at(report['output'],offset+relative,720)
                result['compilation_picture_mean_abs_error']=single_view_error(source,clip,when,combined)['picture_mean_abs_error']
            else:
                src=frame_at(source['path'],when,720)
                target=cv2.resize(src,(720,405),interpolation=cv2.INTER_AREA)
                result={'overview_mean_abs_error':float(np.mean(abs(output[910:1272].astype(float)-target[35:-8].astype(float))))}
            samples.append({'relative_sec':relative,'source_sec':when,**result})
        audio=[]
        if source['has_audio']:
            for relative in (.5,max(.5,clip['duration_sec']-2.5)):
                original=audio_at(source['path'],clip['source_start_sec']+relative)
                individual=audio_alignment(original,audio_at(clip['path'],relative))
                combined=audio_alignment(original,audio_at(report['output'],offset+relative))
                audio.append({'relative_sec':relative,'individual':individual,'compilation':combined})
        if any(value>12 for s in samples for key,value in s.items() if key.endswith('mean_abs_error')):
            raise ValueError(f'源画面匹配失败：{clip["rank"]}')
        for a in audio:
            for kind in ('individual','compilation'):
                d=a[kind]
                if abs(d['lag_ms'])>80 or (d['correlation'] is not None and d['correlation']<.75):
                    raise ValueError(f'原声匹配失败：{clip["rank"]}, {kind}, {d}')
        results.append({'rank':clip['rank'],'video_samples':samples,'audio_samples':audio});offset+=clip['duration_sec']
    effects=[]
    for segment in report.get('segments',[]):
        if segment['kind'] not in ('replay','teaser'):continue
        source=source_for(manifest,segment['rally_id'])
        samples=[]
        for relative in (segment['duration_sec']*.25,segment['duration_sec']*.65):
            when=segment['source_start_sec']+relative*segment['playback_rate']
            combined=frame_at(report['output'],segment['timeline_start_sec']+relative,720)
            clip=by_rank[segment['rank']]
            if clip.get('view_layout')=='single':
                result=single_view_error(source,clip,when,combined,segment['kind'])
                error=result['picture_mean_abs_error']
            else:
                original=frame_at(source['path'],when,720)
                target=cv2.resize(original,(720,405),interpolation=cv2.INTER_AREA)
                # Old files retain their original overview-based verification.
                error=float(np.mean(abs(combined[910:1165].astype(float)-target[35:290].astype(float))))
                result={'overview_mean_abs_error':error}
            if error>12:raise ValueError(f'快切/慢回放源画面或时间映射错误：{segment["index"]}')
            samples.append({'relative_sec':relative,'source_sec':when,**result})
        # The composed audio must match the already time-stretched effect clip.
        # This verifies timeline placement, while the renderer stretches both
        # source streams by the same explicitly recorded playback_rate.
        length=min(1.,segment['duration_sec']-.12)
        a=audio_alignment(audio_at(segment['path'],.06,length),audio_at(report['output'],segment['timeline_start_sec']+.06,length))
        if abs(a['lag_ms'])>80 or (a['correlation'] is not None and a['correlation']<.75):
            raise ValueError('快切/回放在合集中的原声偏移')
        effects.append({'kind':segment['kind'],'rank':segment['rank'],'playback_rate':segment['playback_rate'],
                        'video_samples':samples,'compilation_audio':a})
    method='recorded single-view source crop comparison (within one 30 Hz frame) and source audio cross-correlation' if all(c.get('view_layout')=='single' for c in report['clips']) else 'source overview pixel comparison and source audio cross-correlation'
    save_json(directory/f'alignment_verification{suffix}.json',{'status':'passed','method':method,
              'video_samples_per_clip':3,'audio_windows_per_clip':2,'results':results,'effects':effects})
    print('源画面与原声内容对齐校验通过')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--style',choices=['classic','lively'],default='classic')
    args=p.parse_args();check(args.run.resolve(),args.style)
