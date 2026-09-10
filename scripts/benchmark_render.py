"""Compare sequential/parallel rendering on identical inputs and encoded output."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from volleymole.common import digest, read_json, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=read_json(args.reference/'match_manifest.json')
    decision=read_json(args.reference/'edit_decision.json')
    font=read_json(args.reference/'render_report_lively.json')['font_path']
    selected={r['rally_id'] for r in decision['selected']}
    files=['match_manifest.json','edit_decision.json']
    for rally in manifest['rallies']:
        if rally['rally_id'] in selected:
            files += [rally['tracking_json'],*rally['preview_frames']]
    rows=[]
    for workers in (1,2):
        out=args.output/f'workers-{workers}'
        out.mkdir()  # Never overwrite a previous benchmark.
        for name in files:
            target=out/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(args.reference/name,target)
        command=[sys.executable,'-m','volleymole.media_worker','render','--run',str(out),
                 '--font',font,'--style','lively','--workers',str(workers)]
        started=time.perf_counter()
        with (out/'render.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=read_json(out/'render_report_lively.json')
        row={'workers':workers,'command_wall_sec':time.perf_counter()-started,
             'render_timing_sec':report['render_timing_sec'],
             'output_sha256':digest(report['output'])}
        if workers==2:
            baseline=read_json(args.output/'workers-1/render_report_lively.json')
            row['all_segments_byte_identical']=all(digest(a['path'])==digest(b['path'])
                for a,b in zip(baseline['segments'],report['segments']))
            row['output_byte_identical']=row['output_sha256']==rows[0]['output_sha256']
        rows.append(row)
        save_json(args.output/'comparison.json',rows)
        print(row,flush=True)
    if not rows[-1]['all_segments_byte_identical'] or not rows[-1]['output_byte_identical']:
        raise RuntimeError('Encoded output differs; inspect decoded frames before accepting')


if __name__=='__main__':main()
