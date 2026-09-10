"""Compare real evidence and boundaries; baseline predictions are not human ground truth."""
import argparse
import csv
from itertools import zip_longest
import json
import math
from pathlib import Path
from collections import Counter

from volleymole.common import read_json, save_json


def compare_rows(baseline, candidate):
    count = equal = state_equal = 0
    max_xy = max_pts = 0.
    examples = []
    action_counts = [Counter(),Counter()]
    with (baseline/'tracking/ball.csv').open() as a, (candidate/'tracking/ball.csv').open() as b, \
            (baseline/'tracking/source_pts.csv').open() as ap, (candidate/'tracking/source_pts.csv').open() as bp, \
            (baseline/'analytics/detections.jsonl').open() as ar, (candidate/'analytics/detections.jsonl').open() as br:
        for rows in zip_longest(csv.DictReader(a),csv.DictReader(b),ap,bp,ar,br):
            if any(row is None for row in rows):
                raise ValueError('Inference/PTS frame coverage differs between compared runs')
            x,y,ta,tb,ra,rb = rows
            ra,rb = json.loads(ra),json.loads(rb)
            if int(x['Frame'])!=count or int(y['Frame'])!=count or ra['frame']!=count or rb['frame']!=count:
                raise ValueError(f'Non-contiguous evidence at frame {count}')
            max_pts = max(max_pts,abs(float(ta)-float(tb)),
                          abs(rb['source_time_s']-float(tb)),abs(ra['source_time_s']-float(ta)))
            fields_equal = all(x[k]==y[k] for k in ('Visibility','X','Y','Radius'))
            equal += fields_equal
            if not fields_equal and len(examples)<30:
                examples.append({'frame':count,'baseline':x,'candidate':y})
            if x['Visibility']==y['Visibility']=='1':
                max_xy = max(max_xy, math.hypot(float(x['X'])-float(y['X']),float(x['Y'])-float(y['Y'])))
            state_equal += ra['state']==rb['state']
            for counter,row in zip(action_counts,(ra,rb)):
                counter.update(d['class'] for d in row.get('actions',[]))
            count += 1
    return {'frames':count,'max_pts_error_sec':max_pts,'ball_fields_identical_frames':equal,
            'ball_identical_ratio':equal/count,'max_visible_ball_distance_pixels':max_xy,
            'state_label_agreement':state_equal/count,'action_detection_counts':[dict(c) for c in action_counts],
            'first_ball_differences':examples,'full_coverage':'passed'}


def compare_boundaries(baseline, candidate):
    old = read_json(baseline/'match_manifest.json')
    new = read_json(candidate/'match_manifest.json')
    comparisons = []
    for rally in old['rallies']:
        if not rally['eligible']:
            continue
        def overlap(other):
            return max(0,min(rally['end_sec'],other['end_sec'])-max(rally['start_sec'],other['start_sec']))
        best = max(new['rallies'],key=overlap) if new['rallies'] else None
        comparisons.append({'baseline_id':rally['rally_id'],'baseline_sec':[rally['start_sec'],rally['end_sec']],
            'candidate_id':best['rally_id'] if best else None,
            'candidate_sec':[best['start_sec'],best['end_sec']] if best else None,
            'candidate_eligible':best['eligible'] if best else False,
            'baseline_interval_covered_ratio':overlap(best)/rally['duration_sec'] if best else 0})
    return {'baseline_diagnostics':old['diagnostics'],'candidate_diagnostics':new['diagnostics'],
            'reference_note':'Automated baseline intervals, not human rally truth; overlap is not semantic accuracy.',
            'eligible_baseline_comparisons':comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    evidence = compare_rows(args.baseline,args.candidate)
    result = {'status':'passed_coverage' if evidence['max_pts_error_sec']<.002 else 'failed_pts',
              'evidence':evidence,'boundaries':compare_boundaries(args.baseline,args.candidate)}
    save_json(args.output,result)
    print(json.dumps({k:v for k,v in evidence.items() if k!='first_ball_differences'},indent=2))
    if result['status']!='passed_coverage':
        raise SystemExit(1)


if __name__=='__main__':
    main()
