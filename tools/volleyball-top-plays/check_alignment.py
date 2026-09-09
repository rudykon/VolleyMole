"""Independent content checks against the source, beyond stream duration checks."""
import argparse
from pathlib import Path
import subprocess
import cv2
import numpy as np
from scipy.signal import correlate, correlation_lags
from common import read_json, save_json
from media_worker import frame_at


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


def check(directory,style='classic'):
    suffix='_lively' if style=='lively' else ''
    source=read_json(directory/'match_manifest.json')['source'];report=read_json(directory/f'render_report{suffix}.json')
    results=[];offset=0.
    for clip in report['clips']:
        offset=clip.get('timeline_start_sec',offset)
        samples=[]
        for relative in (.5,clip['duration_sec']/2,clip['duration_sec']-1.):
            src=frame_at(source['path'],clip['source_start_sec']+relative,720)
            output=frame_at(clip['path'],relative,720)
            overview=output[875:1280]
            target=cv2.resize(src,(720,405),interpolation=cv2.INTER_AREA)
            error=float(np.mean(abs(overview[35:-8].astype(float)-target[35:-8].astype(float))))
            samples.append({'relative_sec':relative,'source_sec':clip['source_start_sec']+relative,'overview_mean_abs_error':error})
        audio=[]
        if source['has_audio']:
            for relative in (.5,max(.5,clip['duration_sec']-2.5)):
                original=audio_at(source['path'],clip['source_start_sec']+relative)
                individual=audio_alignment(original,audio_at(clip['path'],relative))
                combined=audio_alignment(original,audio_at(report['output'],offset+relative))
                audio.append({'relative_sec':relative,'individual':individual,'compilation':combined})
        if any(s['overview_mean_abs_error']>12 for s in samples):
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
        samples=[]
        for relative in (segment['duration_sec']*.25,segment['duration_sec']*.65):
            when=segment['source_start_sec']+relative*segment['playback_rate']
            original=frame_at(source['path'],when,720)
            combined=frame_at(report['output'],segment['timeline_start_sec']+relative,720)
            target=cv2.resize(original,(720,405),interpolation=cv2.INTER_AREA)
            # Exclude the teaser footer as well as the full-court caption.
            error=float(np.mean(abs(combined[910:1165].astype(float)-target[35:290].astype(float))))
            if error>12:raise ValueError(f'快切/慢回放源画面或时间映射错误：{segment["index"]}')
            samples.append({'relative_sec':relative,'source_sec':when,'overview_mean_abs_error':error})
        # The composed audio must match the already time-stretched effect clip.
        # This verifies timeline placement, while the renderer stretches both
        # source streams by the same explicitly recorded playback_rate.
        length=min(1.,segment['duration_sec']-.12)
        a=audio_alignment(audio_at(segment['path'],.06,length),audio_at(report['output'],segment['timeline_start_sec']+.06,length))
        if abs(a['lag_ms'])>80 or (a['correlation'] is not None and a['correlation']<.75):
            raise ValueError('快切/回放在合集中的原声偏移')
        effects.append({'kind':segment['kind'],'rank':segment['rank'],'playback_rate':segment['playback_rate'],
                        'video_samples':samples,'compilation_audio':a})
    save_json(directory/f'alignment_verification{suffix}.json',{'status':'passed','method':'source overview pixel comparison and source audio cross-correlation',
              'video_samples_per_clip':3,'audio_windows_per_clip':2,'results':results,'effects':effects})
    print('源画面与原声内容对齐校验通过')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--style',choices=['classic','lively'],default='classic')
    args=p.parse_args();check(args.run.resolve(),args.style)
