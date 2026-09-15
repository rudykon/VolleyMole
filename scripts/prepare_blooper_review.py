#!/usr/bin/env python3
"""Prepare continuous-context clips and PTS-indexed sheets for local review."""
import argparse
import concurrent.futures
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from volleymole.common import read_json,save_json,digest,probe
from volleymole.semantic import sampled_evidence
from PIL import Image,ImageDraw
import base64


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=read_json(a.input);source=probe(data['source']['path']);source['identity']={'sha256':digest(source['path'])}
    if source['identity']['sha256']!=data['source']['sha256']:raise ValueError('source hash mismatch')
    a.output.mkdir(parents=True,exist_ok=True)
    def prepare(pair):
        i,c=pair;start=max(0.,c['center_sec']-12);end=min(source['duration_sec'],c['center_sec']+8)
        folder=a.output/f'candidate_{i+1:02d}';folder.mkdir(exist_ok=True)
        clip=folder/'context.mp4'
        if not clip.exists():
            subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-i',source['path'],'-t',str(end-start),
                '-map','0:v:0','-map','0:a:0?','-vf','scale=960:-2','-c:v','libx264','-preset','veryfast','-crf','21',
                '-threads','2','-c:a','aac','-b:a','128k','-movflags','+faststart',str(clip)],check=True,timeout=90)
        evidence,content,_=sampled_evidence(source,start,end,2,480,None,time.monotonic()+90)
        frames=[Image.open(BytesIO(base64.b64decode(x['image_url']['url'].split(',',1)[1]))) for x in content if x['type']=='image_url']
        pages=[]
        for page,offset in enumerate(range(0,len(frames),20)):
            selected=frames[offset:offset+20];sheet=Image.new('RGB',(480*4,294*((len(selected)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
            for j,frame in enumerate(selected):
                x,y=(j%4)*480,(j//4)*294;sheet.paste(frame,(x,y+24))
                ev=evidence[offset+j];draw.text((x+4,y+4),f"C{i+1:02d} {ev['id']} {ev['start_sec']:.3f}s",fill='black')
            path=folder/f'sheet_{page+1}.jpg';sheet.save(path,quality=92);pages.append(str(path))
        info={'id':f'candidate_{i+1:02d}','source':source,'start_sec':start,'end_sec':end,
            'center_sec':c['center_sec'],'audio_candidate':c,'clip':str(clip),'clip_sha256':digest(clip),
            'clip_duration_sec':probe(clip)['duration_sec'],'sampling_fps':2,'frames':evidence,'sheets':pages,
            'review_status':'prepared','annotation_source':'model','is_ground_truth':False}
        save_json(folder/'evidence.json',info);print(info['id'],len(frames),flush=True);return info
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:items=list(pool.map(prepare,enumerate(data['candidates'])))
    save_json(a.output/'evidence.json',{'source_sha256':source['identity']['sha256'],'items':items,
        'source_candidates_sha256':digest(a.input),'note':'Local review evidence; preparation is not a finding.'})


if __name__=='__main__':main()
