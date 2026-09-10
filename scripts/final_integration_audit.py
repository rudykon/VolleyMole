"""Recheck final local evidence; fail rather than mark missing real runs as passed."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from volleymole.common import APP, digest, identity, read_json, save_json
from volleymole.models import ModelRegistry
from volleymole.schemas import validate_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--top5-folder',default='final-qwen-top5')
    args = parser.parse_args()
    root = args.root.resolve()
    integration = root/'runs/integration'
    checks = []
    for label,command in (
        ('final-unit-tests',[sys.executable,'-m','unittest','discover','-s','tests','-v']),
        ('final-legacy-tests',[sys.executable,'-m','unittest','discover','-s','tools/volleyball-top-plays/tests','-v']),
        ('final-diff-check',['git','diff','--check'])):
        begin = time.monotonic()
        with (integration/(label+'.log')).open('w') as log:
            result = subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{label} failed; inspect its log')
        checks.append({'name':label,'returncode':0,'elapsed_sec':time.monotonic()-begin})
    videos = []
    for folder,k in [(args.top5_folder,5),('final-wheel-top10-api-failure',10)]:
        directory = integration/folder
        manifest,decision = (read_json(directory/name) for name in ('match_manifest.json','edit_decision.json'))
        validate_decision(decision,manifest,directory,k)
        if k==5 and decision['ranking_mode'] not in ('multimodal_api','vision_then_text_api'):
            raise RuntimeError('Final semantic top5 did not succeed; preserve fallback evidence separately')
        for name in ('verification_lively.json','alignment_verification_lively.json','title_card_readability.json'):
            if read_json(directory/name)['status']!='passed':
                raise RuntimeError(f'{folder}/{name} did not pass')
        render = read_json(directory/'render_report_lively.json')
        video = identity(Path(render['output']))
        recorded = read_json(directory/'verification_lively.json')['videos'][-1]
        if video['sha256']!=recorded['sha256']:
            raise RuntimeError('Verified video hash changed')
        videos.append({'top_k':k,'video':video,'duration_sec':render['expected_duration_sec'],
            'ranking_mode':decision['ranking_mode'],'fallback':decision['fallback'],
            'selected':decision['selected'],'timing':read_json(directory/'timing_lively.json')})
    for path in ('phase3-full-match/real-resume-regression.json',
                 args.top5_folder+'/real-resume-regression.json',
                 'final-wheel-top10-api-failure/api-failure-regression.json',
                 'final-wheel-release-smoke/installed-package-proof.json'):
        if read_json(integration/path)['status']!='passed':
            raise RuntimeError(f'{path} did not pass')
    if read_json(integration/'final-classic-silent-fixture/verification.json')['status']!='passed':
        raise RuntimeError('Classic/silent/missing-track regression did not pass')
    comparison = read_json(integration/'phase3-baseline-comparison.json')
    rows = comparison['boundaries']['eligible_baseline_comparisons']
    if not all(r['candidate_eligible'] and r['baseline_interval_covered_ratio']==1 for r in rows):
        raise RuntimeError('A baseline eligible interval was lost')
    if len({r['candidate_id'] for r in rows})!=len(rows):
        raise RuntimeError('Previously separate baseline intervals were merged')
    tracked = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    forbidden = {'.pt','.pth','.onnx','.safetensors','.mp4','.mov','.avi'}
    bad = [p for p in tracked if p and (Path(p).suffix.lower() in forbidden or
           Path(p).name in ('llm_api.json','github_token.json') or Path(p).parts[0] in ('models','runs','data','.venv'))]
    if bad:
        raise RuntimeError('Local-only files are visible to Git: '+repr(bad))
    wheel = root/'dist/volleymole-0.2.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        if any(Path(p).suffix.lower() in forbidden for p in archive.namelist()):
            raise RuntimeError('Wheel contains model weights or match media')
        for source in APP.glob('*.py'):
            if archive.read('volleymole/'+source.name)!=source.read_bytes():
                raise RuntimeError('Wheel code is stale: '+source.name)
    phase2 = read_json(integration/'phase2-full-summary.json')
    report = {'status':'passed_with_documented_limits','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'checks':checks,'models':ModelRegistry(root/'models').verify(),'wheel':identity(wheel),
        'code_sha256':{p.name:digest(p) for p in APP.glob('*.py')},'videos':videos,
        'baseline_eligible_intervals_preserved_separately':len(rows),
        'full_inference':{k:phase2[k] for k in ('counts','inference_wall_sec','command_wall_sec',
            'torch_peak_allocated_bytes','sampled_devices','backend','benchmark_caveat')},
        'quality_diagnostics':comparison['boundaries']['candidate_diagnostics'],
        'limits':['No independently human-labelled full-match ground truth or true jersey-number accuracy score.',
                  'The rules-fallback top10 includes a multi-ball warmup interval at rank 8; playback/boundary checks are not perfect semantic selection.',
                  'Core/person/OCR remote weight downloads failed TLS or transfer checks; all nine local files verified; public VballNet fetch passed.',
                  'One CUDA 0 inference run, not four-GPU execution; device memory includes unowned activity, and historical four-card costs are not equal-hardware benchmarks.',
                  'Only Linux x86-64 / Python 3.12 / this RTX 3090 driver environment was tested.',
                  'Configured GLM final-ranking attempts timed out; capped Qwen final ranking passed with explicit CLI overrides. Default configuration was not edited; capped GLM was not retested.',
                  'Third-party source notices do not grant blanket redistribution rights for dependencies or weights.'],
        'preservation':'Original video, models, source logo, upstream checkouts and prior completed videos retained; no commit or push performed.'}
    save_json(integration/'final-integration-audit.json',report)
    print('Final integration audit:',report['status'])


if __name__=='__main__':
    main()
