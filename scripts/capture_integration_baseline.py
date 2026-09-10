"""Freeze the actual working tree and re-run legacy checks without copying secrets.

This is an explicit baseline operation; archives/runs/weights stay outside Git.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'tools/volleyball-top-plays'


def identity(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return {'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes':path.stat().st_size,'sha256':digest.hexdigest()}


def command(argv,directory,name):
    start=time.monotonic()
    with (directory/f'{name}.log').open('w') as log:
        result=subprocess.run([str(arg) for arg in argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    return {'command':[str(arg) for arg in argv],'returncode':result.returncode,
            'elapsed_sec':round(time.monotonic()-start,3),'log':f'{name}.log'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'runs/1-top5')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();output=args.output.resolve();run=args.run.resolve()
    output.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat()
    # Explicit source/document allowlist: never copy API configs, credentials,
    # third-party virtualenvs, hidden Codex data or arbitrary project-root files.
    files=[ROOT/'README.md',ROOT/'.gitignore',ROOT/'volleymole.svg']
    files += [p for p in APP.rglob('*') if p.is_file() and '__pycache__' not in p.parts
              and '.venv' not in p.parts and p.suffix in ('.py','.md','.json','.txt','.png','.ttf')]
    files += list((ROOT/'docs').glob('*.md'))
    files += [ROOT/'tools/PINNED_UPSTREAM.md']
    files=sorted(set(p for p in files if p.is_file()))
    with tarfile.open(output/'working-source.tar.gz','w:gz') as archive:
        for path in files:archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    source_manifest=[identity(p) for p in files]
    tracking_python=ROOT/'tools/fast-volleyball-tracking-inference/.venv/bin/python'
    checks=[command([tracking_python,'-m','unittest','discover','-s',APP/'tests','-v'],output,'unit-tests')]
    if args.verify:
        checks.append(command([sys.executable,APP/'run_match.py','--video',ROOT/'data/样例视频/1.mp4',
                               '--top-k','5','--style','lively','--rerun-from','verify'],output,'real-top5-verification'))
    records=output/'records';records.mkdir()
    record_names=('match_manifest.json','edit_decision.json','render_report_lively.json',
                  'verification_lively.json','alignment_verification_lively.json','title_card_readability.json',
                  'timing_lively.json','timing_latest.json','run_config.json')
    for name in record_names:
        if (run/name).is_file():shutil.copy2(run/name,records/name)
    report=json.loads((run/'render_report_lively.json').read_text())
    video=Path(report['output']);shutil.copy2(video,output/video.name)
    manifest=json.loads((run/'match_manifest.json').read_text())
    decision=json.loads((run/'edit_decision.json').read_text())
    gpu=command(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version',
                 '--format=csv,noheader'],output,'gpu-snapshot')
    environments={}
    probe_code='import json,sys,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}},sort_keys=True))'
    for name,python in [('analytics',ROOT/'tools/volleyball_analytics/.venv-inference/bin/python'),
                        ('tracking',tracking_python),('numbers',ROOT/'tools/volleyball-highlights/.venv/bin/python')]:
        environments[name]=json.loads(subprocess.check_output([str(python),'-c',probe_code],text=True))
    upstreams={}
    for name,folder in [('analytics',ROOT/'tools/volleyball_analytics'),
                        ('ml_core',ROOT/'tools/volleyball_analytics/src/ml_manager'),
                        ('vballnet',ROOT/'tools/fast-volleyball-tracking-inference'),
                        ('numbers',ROOT/'tools/volleyball-highlights')]:
        revision=subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],text=True,capture_output=True)
        upstreams[name]={'revision':revision.stdout.strip() if revision.returncode==0 else None,
                         'license_present':(folder/'LICENSE').is_file()}
    data={'captured_at_utc':started,'status':'passed' if all(c['returncode']==0 for c in checks) else 'failed',
          'source_archive':'working-source.tar.gz','working_source':source_manifest,'checks':checks,'gpu_snapshot':gpu,
          'environments':environments,'upstreams':upstreams,'source_video':identity(Path(manifest['source']['path'])),
          'output_video':identity(video),'output_copy':video.name,'design_revision':report['design_revision'],
          'source_metadata':manifest['source'],'rally_diagnostics':manifest['diagnostics'],
          'selected':decision['selected'],'ranking_mode':decision['ranking_mode'],
          'baseline_scope':'Existing real top5 with cached analysis; full current decode/audio/title/source checks rerun if --verify. Historical GPU inference costs are not a new end-to-end benchmark.',
          'known_gaps':['No independently human-labeled full-match rally ground truth.',
                        'Existing real top5 uses cached inference; real top10 and new unified fresh-inference gates remain required.']}
    (output/'baseline.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':data['status'],'output':str(output),'checks':checks},ensure_ascii=False,indent=2))
    if data['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
