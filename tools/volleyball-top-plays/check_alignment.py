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


def check(directory):
    source=read_json(directory/'match_manifest.json')['source'];report=read_json(directory/'render_report.json')
    results=[];offset=0.
    for clip in report['clips']:
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
    save_json(directory/'alignment_verification.json',{'status':'passed','method':'source overview pixel comparison and source audio cross-correlation',
              'video_samples_per_clip':3,'audio_windows_per_clip':2,'results':results})
    print('源画面与原声内容对齐校验通过')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    check(p.parse_args().run.resolve())
