"""Sequential real-GPU ablations. Keeps original outputs and fails on evidence drift."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from volleymole.common import digest, read_json, save_json


RAW = ('analytics/detections.jsonl','tracking/ball.csv','tracking/source_pts.csv','player/index.json')


def compare(reference, candidate):
    checks={name:digest(reference/name)==digest(candidate/name) for name in RAW}
    a,b=(read_json(p/'summary.json') for p in (reference,candidate))
    checks.update(counts=a['counts']==b['counts'],state_windows=a['state_windows']==b['state_windows'])
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--models',type=Path,default=Path('models'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-frames',type=int,default=902)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    variants=[('reference',[]),('pipeline',['--pipeline-depth','2']),
              ('auxiliary',['--pipeline-depth','2','--auxiliary-device','cuda:0']),
              ('bound',['--pipeline-depth','2','--auxiliary-device','cuda:0','--vball-engine','ort-bound'])]
    records=[]
    for name,options in variants:
        out=args.output/name
        if out.exists():raise FileExistsError(out)
        command=[sys.executable,'-m','volleymole.inference','--kind','shared','--video',str(args.video),
                 '--models',str(args.models),'--output',str(out),'--half',
                 '--devices','cuda:0,cuda:1,cuda:2,cuda:3','--max-frames',str(args.max_frames),*options]
        start=time.perf_counter()
        with (args.output/f'{name}.log').open('w') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'variant':name,'command':command,'returncode':completed.returncode,
             'command_wall_sec':time.perf_counter()-start}
        if completed.returncode==0:
            row['inference_sec']=read_json(out/'telemetry.json')['wall_sec']
            row['checks']=compare(args.output/'reference',out)
        records.append(row)
        save_json(args.output/'comparison.json',records)
        print(json.dumps(row),flush=True)
        if completed.returncode or not all(row['checks'].values()):
            raise RuntimeError(f'{name} failed; inspect {args.output}')


if __name__=='__main__':main()
