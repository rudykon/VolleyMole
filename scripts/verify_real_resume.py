"""Actual full-video cache reuse and recovery of a missing verification artifact."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from volleymole.common import read_json, save_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--top-k',type=int,choices=[5,10],required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--analysis-cache-dir',type=Path,required=True)
    parser.add_argument('--llm-config',type=Path,required=True)
    parser.add_argument('--model')
    parser.add_argument('--vision-model')
    args = parser.parse_args()
    directory = args.run.resolve()
    output = directory/f'top{args.top_k}_lively.mp4'
    original_sha = digest(output)
    command = [sys.executable,'-m','volleymole','run','--video',str(args.video.resolve()),
        '--top-k',str(args.top_k),'--output',str(directory),'--models',str(args.models.resolve()),
        '--analysis-cache-dir',str(args.analysis_cache_dir.resolve()),'--device','cuda:0',
        '--llm-config',str(args.llm_config.resolve()),'--ranker','auto','--api-timeout','120']
    if args.model:
        command += ['--model',args.model]
    if args.vision_model:
        command += ['--vision-model',args.vision_model]
    def run(label):
        begin = time.monotonic()
        with (directory/f'resume-{label}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        timing = read_json(directory/'timing_latest.json')
        if digest(output)!=original_sha:
            raise RuntimeError('Resume unexpectedly changed the finished video')
        return {'elapsed_sec':time.monotonic()-begin,'timing':timing}
    full = run('unchanged')
    if any(s['status']!='reused' for s in full['timing']['stages']):
        raise RuntimeError('Unchanged real run did not reuse every stage')
    artifact = directory/'verification_lively.json'
    backup = directory/'verification_lively.resume-backup.json'
    if backup.exists():
        raise FileExistsError('Previous recovery backup exists; inspect before repeating test')
    artifact.replace(backup)
    try:
        recovered = run('missing-verification')
    finally:
        # The valid historical report stays recoverable even if the test fails.
        if not artifact.exists():
            backup.replace(artifact)
    stages = recovered['timing']['stages']
    if any(s['status']!='reused' for s in stages if s['stage']!='verify_lively'):
        raise RuntimeError('Artifact recovery reran unrelated stages')
    if next(s for s in stages if s['stage']=='verify_lively')['status']!='complete':
        raise RuntimeError('Missing verification artifact was not regenerated')
    save_json(directory/'real-resume-regression.json',{'status':'passed','output_sha256':original_sha,
        'unchanged_run':full,'missing_artifact_recovery':recovered,
        'backup':str(backup),'mechanism':'Real completed run, then recoverably moved verification JSON; no inference or API mocks.'})
    print('Real resume and missing-artifact recovery passed')


if __name__=='__main__':
    main()
