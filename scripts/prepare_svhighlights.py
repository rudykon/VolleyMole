#!/usr/bin/env python3
"""Acquire existing SVHighlights labels without regenerating annotations."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPOSITORY = 'idong1004/SVHighlights'
REVISION = 'fdedf750ddc2524093eda9bca1fbe24d1210724a'
FILES = {
    'README.md': 'd803b3860c7457cf2fcd37fd91b4b60832745135ce5c7558299187fa0e0bf633',
    'annotations/label.json': 'dea5697c5a736a9c7c2a926528ade204d6c75a56f1b9f8862808a76d763992f3',
    'metadata/video_list.csv': 'b109e57c803eab639e9d3abb156fca04c863a0e403cbd00d89ea1742039a7c11',
    'annotations/volume.json': '121dca3f5d0a686e2db99b8894fab1089faa4d126b8448b6f50e4b40cf61715b',
    'annotations/minmax_volume.json': '4aa8e0220960c5ef9de4842c3722c34e375525af66d65fc385f9838525fc2066',
}


def prepare(directory):
    artifacts=[]
    for name,expected in FILES.items():
        path=directory/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Pinned source checksum mismatch: {name}')
        artifacts.append({'path':name,'sha256':actual,'bytes':path.stat().st_size})
    labels=json.loads((directory/'annotations/label.json').read_text())
    volleyball=[r for r in labels if r['vid'].startswith('volleyball_')]
    if len(volleyball)!=40 or len({r['vid'] for r in volleyball})!=40:
        raise ValueError('Expected all 40 original volleyball broadcasts')
    with (directory/'metadata/video_list.csv').open(newline='') as stream:
        metadata={r['vid']:r for r in csv.DictReader(stream)}
    videos=[]
    for row in volleyball:
        if any(type(x) is not int or x not in (0,1) for x in row['saliency_scores']):
            raise ValueError('Original highlight labels must remain binary')
        meta=metadata[row['vid']]
        videos.append({'vid':row['vid'],'full_url':meta['full_link'],'official_highlight_url':meta['hl_link'],
            'source_trim_start_sec':float(meta['full_start']),'source_trim_end_sec':float(meta['full_end']),
            'label_bin_sec':2,'label_bins':len(row['saliency_scores']),'positive_bins':sum(row['saliency_scores'])})
    # Exact subset of released rows; neither new human labels nor pseudo-labels.
    (directory/'volleyball_labels.json').write_text(json.dumps(volleyball,ensure_ascii=False)+'\n')
    manifest={'dataset':'SVHighlights','hub_repository':REPOSITORY,'revision':REVISION,
        'source_url':f'https://huggingface.co/datasets/{REPOSITORY}',
        'project_url':'https://github.com/leedongkyu2019/SVHighlights',
        'license':'CC-BY-NC-4.0 for annotations/features; original broadcasts retain publisher terms',
        'annotation_origin':'author_released_official_highlight_alignment_with_author_filtering',
        'ground_truth_scope':'editorial highlight proxy, not independently human-rated rally quality or blooper humor',
        'training_or_new_annotation_performed':False,'acquired_utc':datetime.now(timezone.utc).isoformat(),
        'artifacts':artifacts,'videos':videos,'original_video_media_downloaded':False,
        'splits':'No train/validation/test partition is invented; use as an evaluation-only collection.',
        'label_time_basis':'2-second bins in the authors trimmed full video; subtract source_trim_start_sec from untrimmed video timestamps',
        'unavailable_metrics':['human top-five quality','explanation correctness','blooper ranking','complete rally edit boundaries']}
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/public_benchmarks/svhighlights'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--endpoint',default='https://huggingface.co')
    parser.add_argument('--direct',action='store_true',help='Ignore proxy variables for this public download only')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.prepare_only:
        hf=shutil.which('hf') or str(Path(sys.executable).with_name('hf'))
        environment=os.environ.copy()
        environment.update(HF_ENDPOINT=args.endpoint,HF_HOME=str(args.output.resolve()/'.hf'),
                           HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
        if args.direct:
            for key in ('HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy'):environment.pop(key,None)
        subprocess.run([hf,'download',REPOSITORY,'--repo-type','dataset','--revision',REVISION,
            '--include',*FILES,'--local-dir',str(args.output),'--max-workers','1'],env=environment,check=True)
    manifest=prepare(args.output)
    print(json.dumps({'videos':len(manifest['videos']),'positive_bins':sum(v['positive_bins'] for v in manifest['videos']),
                      'manifest':str(args.output/'manifest.json')},ensure_ascii=False))


if __name__=='__main__':main()
