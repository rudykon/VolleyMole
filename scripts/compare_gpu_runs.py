"""Record a real GPU snapshot or compare fresh single/four-GPU full-match runs."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess

from volleymole.common import digest, read_json, save_json


def snapshot():
    result = {'at_utc': datetime.now(timezone.utc).isoformat()}
    for key, query in (
            ('devices', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu'),
            ('compute_apps', '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory')):
        text = subprocess.check_output(['nvidia-smi', query, '--format=csv,nounits'], text=True, timeout=10)
        result[key] = [{k.strip(): v.strip() for k,v in row.items()}
                       for row in csv.DictReader(io.StringIO(text))]
    return result


def compare_smoke(baseline, candidate):
    from compare_integration_runs import compare_rows
    raw = compare_rows(baseline,candidate)
    hashes = {name: digest(baseline/name) == digest(candidate/name) for name in (
        'analytics/detections.jsonl','tracking/ball.csv','tracking/source_pts.csv','player/index.json')}
    a,b = (read_json(run/'provenance.json') for run in (baseline,candidate))
    sa,sb = (read_json(run/'summary.json') for run in (baseline,candidate))
    checks = {'same_source': all(a['source'][k] == b['source'][k] for k in ('sha256','bytes')),
              'same_frame_limit': a['parameters']['max_frames'] == b['parameters']['max_frames'],
              'same_frame_count': raw['frames'] == sa['processed_frames'] == sb['processed_frames'],
              'same_inference_counts': sa['counts'] == sb['counts'],
              'byte_identical_evidence_and_ocr': all(hashes.values())}
    return {'status':'passed' if all(checks.values()) else 'differences_require_review',
            'checks':checks,'identical_files':hashes,'evidence':raw,
            'scope':'Short source or explicitly partial smoke, not a full-match benchmark.'}


def compare(baseline, candidate, manifest_baseline):
    # This command is intentionally run after inference, not concurrently with
    # it: streaming both detection files would contaminate a wall-time trial.
    from compare_integration_runs import compare_rows, compare_boundaries
    old = read_json(baseline/'inference_cache.json')
    new = read_json(candidate/'inference_cache.json')
    old_timing = read_json(baseline/'timing_latest.json')
    new_timing = read_json(candidate/'timing_latest.json')
    old_manifest = read_json(manifest_baseline/'match_manifest.json')
    new_manifest = read_json(candidate/'match_manifest.json')
    raw = compare_rows(baseline, candidate)
    hashes = {name: {'single': digest(baseline/name), 'four': digest(candidate/name)}
              for name in ('analytics/detections.jsonl','tracking/ball.csv','tracking/source_pts.csv')}
    for row in hashes.values():
        row['identical'] = row['single'] == row['four']
    fields = ('rally_id','start_sec','end_sec','eligible')
    intervals_equal = ([{k:r[k] for k in fields} for r in old_manifest['rallies']] ==
                       [{k:r[k] for k in fields} for r in new_manifest['rallies']])
    source_equal = all(old_manifest['source']['identity'][k] == new_manifest['source']['identity'][k]
                       for k in ('sha256','bytes'))
    def models(run):
        provenance = read_json(run/'analytics/provenance.json')
        while 'original' in provenance:
            provenance = provenance['original']
        return {k:v['sha256'] for k,v in provenance['models'].items()}
    execution = new['summary']['gpu_execution']
    telemetry = new['historical_inference_telemetry']
    actual = execution['actual_model_devices']
    device_stats = {}
    for sample in telemetry['gpu_samples']:
        for index, uuid, name, memory, utilization in sample['devices']:
            row = device_stats.setdefault(index, {'name':name,'uuid':uuid,'peak_memory_mib':0,
                                                  'peak_utilization_percent':0,'utilization_samples':[]})
            row['peak_memory_mib'] = max(row['peak_memory_mib'], int(memory))
            row['peak_utilization_percent'] = max(row['peak_utilization_percent'],int(utilization))
            row['utilization_samples'].append(int(utilization))
    for row in device_stats.values():
        values = row.pop('utilization_samples')
        row['mean_utilization_percent'] = round(sum(values)/len(values),3)
    checks = {'same_source_bytes': source_equal,
              'same_model_hashes': models(baseline) == models(candidate),
              'fresh_single_gpu_baseline': old_timing['analysis_cache']['fresh_inference_this_command'],
              'fresh_four_gpu_inference': new_timing['analysis_cache']['fresh_inference_this_command'],
              'full_coverage': raw['frames'] == new['summary']['processed_frames'] == old['summary']['processed_frames'],
              'four_distinct_actual_devices': len(set(actual.values())) == 4 and actual == execution['role_devices'],
              'real_ort_cuda_kernels': new['summary']['backend']['kernel_provider_counts'].get('CUDAExecutionProvider',0)>0,
              'all_torch_devices_allocated': len(telemetry['torch_devices']) == 4 and all(
                  v['peak_allocated_bytes']>0 for v in telemetry['torch_devices'].values()),
              'raw_evidence_byte_identical': all(row['identical'] for row in hashes.values()),
              'state_windows_identical': old['summary']['state_windows'] == new['summary']['state_windows'],
              'inference_counts_identical': old['summary']['counts'] == new['summary']['counts'],
              'rally_intervals_and_eligibility_identical': intervals_equal,
              'one_decode_pass': new['summary']['counts']['decode_passes'] == 1}
    single_sec = old_timing['total_elapsed_sec']
    four_sec = new_timing['total_elapsed_sec']
    return {'status': 'passed' if all(checks.values()) else 'differences_require_review', 'checks': checks,
            'baseline': str(baseline), 'candidate': str(candidate), 'manifest_baseline':str(manifest_baseline),
            'command_wall_sec': {'single': single_sec, 'four': four_sec,
                'observed_speedup': round(single_sec/four_sec,3),
                'observed_time_reduction_percent': round(100*(1-four_sec/single_sec),2)},
            'inference_body_wall_sec': {'single': old['historical_inference_telemetry']['wall_sec'],
                                        'four':telemetry['wall_sec']},
            'timing_scope': 'Fresh source analysis through rally manifest, including model loading; no LLM calls, previews, rendering or OCR. Historical single-card trial versus current four-card trial, not an isolated or repeated benchmark.',
            'raw_hashes':hashes,'evidence':raw,'boundaries':compare_boundaries(manifest_baseline,candidate),
            'gpu_execution':execution,'gpu_whole_device_samples':device_stats,
            'torch_devices':telemetry['torch_devices'],'inference_pid':telemetry['pid'],
            'memory_note':telemetry['memory_note']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--manifest-baseline',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.snapshot:
        result = snapshot()
    else:
        if not args.baseline or not args.candidate:
            parser.error('--baseline and --candidate are required for comparison')
        result = (compare_smoke(args.baseline,args.candidate) if args.smoke else
                  compare(args.baseline,args.candidate,args.manifest_baseline or args.baseline))
    save_json(args.output,result)
    print(json.dumps(result if args.snapshot or args.smoke else {k:result[k] for k in
                     ('status','checks','command_wall_sec','inference_body_wall_sec')},ensure_ascii=False,indent=2))
    if not args.snapshot and result['status']!='passed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
