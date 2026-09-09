"""Actual rendering, optionally exercising an injected API transport failure."""
import argparse
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from urllib.error import URLError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import APP, probe, read_json, save_json
from ranker import rank, rule_decision


def main(directory,python,api_failure=False):
    directory.mkdir(parents=True,exist_ok=True)
    source=directory/'synthetic.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=30',
                    '-t','25','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True)
    meta=probe(source);rallies=[]
    for i in range(10):
        rid=f'synthetic_{i:02d}';a=i*2+.3;b=i*2+1.7
        name=f'tracking/{rid}.json'
        save_json(directory/name,{'positions':[],'samples':[],'start_frame':round(a*30),'last_frame':round(b*30)-1})
        rallies.append({'rally_id':rid,'start_sec':a,'end_sec':b,'safe_start_sec':a,'safe_end_sec':b,
            'duration_sec':b-a,'eligible':True,'tracking_json':name,'rule_score':100-i,'actions':[],
            'players':[],'exclusion_reasons':[],
            'ball_metrics':{'visible_ratio':0,'trajectory_changes':0},'preview_times_sec':[a,(a+b)/2,b-.05],
            'preview_frames':[f'previews/{rid}_{j}.jpg' for j in range(3)]})
    manifest={'source':meta,'rallies':rallies,'fixture':True,'config':{'preview_limit':25}}
    save_json(directory/'match_manifest.json',manifest)
    subprocess.run([python,str(APP/'media_worker.py'),'previews','--run',str(directory)],check=True)
    if api_failure:
        # Inject only the transport failure; decision validation, fallback,
        # video encoding and full-decode verification run unmocked.
        with patch.dict('os.environ',{'VOLLEYMOLE_API_KEY':'fixture-only-token'}), \
             patch('ranker.request_json',side_effect=URLError('injected offline transport')):
            rank(manifest,directory,10,None,'auto','https://example.invalid/v1','fixture-model')
        decision=read_json(directory/'edit_decision.json')
        if decision['ranking_mode']!='rules_fallback' or decision['fallback']['type']!='URLError':
            raise RuntimeError('API 故障没有进入规则降级')
    else:
        decision=rule_decision(rallies,10);decision['ranking_mode']='synthetic_fixture'
        save_json(directory/'edit_decision.json',decision)
    subprocess.run([python,str(APP/'media_worker.py'),'render','--run',str(directory)],check=True)
    from run_match import verify
    verify(directory,10)
    print('十段实际渲染、无音频、缺失轨迹居中降级：通过'+('；API 故障注入后自动成片：通过' if api_failure else ''))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--python',required=True)
    p.add_argument('--api-failure',action='store_true')
    args=p.parse_args();main(args.output.resolve(),args.python,args.api_failure)
