"""Compare whole-match evidence, selection, preview pixels and output to a baseline."""
import argparse
import json
from pathlib import Path
from volleymole.common import digest, read_json, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--render-reference',type=Path,
                        help='Use a current-code sequential render when historical presentation code differs')
    args=parser.parse_args()
    a,b=args.reference,args.candidate
    checks={name:digest(a/name)==digest(b/name) for name in
            ('analytics/detections.jsonl','tracking/ball.csv','tracking/source_pts.csv','player/index.json')}
    sa,sb=(read_json(p/'analytics/summary.json') for p in (a,b))
    checks['frame_counts_and_calls']=sa['counts']==sb['counts']
    checks['state_windows']=sa['state_windows']==sb['state_windows']
    ma,mb=(read_json(p/'match_manifest.json') for p in (a,b))
    checks['rallies']=ma['rallies']==mb['rallies']
    da,db=(read_json(p/'edit_decision.json') for p in (a,b))
    checks['selected']=da['selected']==db['selected']
    previews=read_json(b/'previews/index.json')['files']
    checks['selected_previews_byte_identical']=all(digest(a/p)==digest(b/p) for p in previews)
    render_reference=args.render_reference or a
    ra,rb=(read_json(p/'render_report_lively.json') for p in (render_reference,b))
    checks['segments_byte_identical']=len(ra['segments'])==len(rb['segments']) and all(
        digest(x['path'])==digest(y['path']) for x,y in zip(ra['segments'],rb['segments']))
    checks['output_byte_identical']=digest(ra['output'])==digest(rb['output'])
    checks['verification_passed']=read_json(b/'verification_lively.json')['status']=='passed'
    result={'reference':str(a.resolve()),'render_reference':str(render_reference.resolve()),
            'candidate':str(b.resolve()),'checks':checks,
            'preview_count':len(previews),'baseline_preview_count':len(read_json(a/'previews/index.json')['files']),
            'timing':read_json(b/'timing_latest.json')}
    name='optimization_comparison_rebased.json' if args.render_reference else 'optimization_comparison.json'
    save_json(b/name,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
