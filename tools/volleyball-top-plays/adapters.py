"""Run upstream projects in their own Python environments, or import validated caches."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
from common import APP, ROOT, identity, read_json, run, save_json

TRACK = ROOT/'tools/fast-volleyball-tracking-inference'
ANALYTICS = ROOT/'tools/volleyball_analytics'
PLAYER = ROOT/'tools/volleyball-highlights'
MODEL = TRACK/'models/VballNetV1_seq9_grayscale_330_h288_w512.onnx'


def project_info(path, models):
    result = subprocess.run(['git','-C',str(path),'rev-parse','HEAD'], capture_output=True, text=True)
    return {'project': path.name, 'path': str(path), 'commit': result.stdout.strip() if result.returncode == 0 else None,
            'models': [identity(p) for p in models if p.is_file()]}


def discover(video, frame_count):
    """Reuse only caches explicitly naming this source and reporting full coverage."""
    found = {}
    for p in sorted((ANALYTICS/'output').glob('*/summary.json'), key=lambda p:p.stat().st_mtime, reverse=True):
        try:
            data = read_json(p)
            if Path(data.get('input','')).resolve() == video and data.get('status') == 'complete' and data.get('processed_frames') == frame_count and (p.parent/'detections.jsonl').is_file():
                found['analytics'] = p.parent
                break
        except (ValueError, OSError, TypeError):
            continue
    for p in sorted((TRACK/'output').glob('*/review_metrics.json'), key=lambda p:p.stat().st_mtime, reverse=True):
        try:
            data = read_json(p)
            if Path(data.get('input','')).resolve() == video and data.get('frames') == frame_count and (p.parent/video.stem/'ball.csv').is_file() and (p.parent/'source_pts.csv').is_file():
                found['tracking'] = p.parent
                break
        except (ValueError, OSError, TypeError):
            continue
    return found


def ingest_analytics(video, directory, cache, python, device):
    out = directory/'analytics'
    out.mkdir(exist_ok=True)
    provenance = project_info(ANALYTICS, list((ANALYTICS/'weights').glob('**/*.pt')) + list((ANALYTICS/'weights/game_state').glob('*.safetensors')))
    if cache:
        cache = Path(cache)
        summary = read_json(cache/'summary.json')
        if Path(summary.get('input','')).resolve() != video:
            raise ValueError('比赛分析缓存指向其他视频')
        provenance.update(mode='imported_cache', raw_summary=identity(cache/'summary.json'), raw_detections=identity(cache/'detections.jsonl'), parameters={k:summary[k] for k in ('imgsz','batch_size','half','prefetch') if k in summary})
        shutil.copyfile(cache/'detections.jsonl', out/'detections.jsonl')
        shutil.copyfile(cache/'summary.json', out/'summary.json')
    else:
        run([python, APP/'upstream_worker.py', 'analytics', '--video',video, '--output',out, '--device',device], out/'inference.log')
        provenance.update(mode='fresh_inference', parameters={'device': device, 'state_window_frames': 30, 'imgsz': 640})
    save_json(out/'provenance.json', provenance)
    return str(out/'detections.jsonl'), [out/'detections.jsonl', out/'summary.json', out/'provenance.json']


def ingest_tracking(video, directory, cache, python, device):
    out = directory/'tracking'
    out.mkdir(exist_ok=True)
    provenance = project_info(TRACK, [MODEL])
    if cache:
        cache = Path(cache)
        summary = read_json(cache/'review_metrics.json')
        if Path(summary.get('input','')).resolve() != video:
            raise ValueError('球追踪缓存指向其他视频')
        raw = cache/video.stem/'ball.csv'
        provenance.update(mode='imported_cache', raw_ball=identity(raw), raw_pts=identity(cache/'source_pts.csv'), parameters={'confidence_threshold':.5})
        shutil.copyfile(raw,out/'ball.csv')
        shutil.copyfile(cache/'source_pts.csv',out/'source_pts.csv')
    else:
        attempt = out/f'upstream_{time.time_ns()}'
        run([python, TRACK/'src/inference_onnx_seq_gray_v2.py', '--video_path',video, '--model_path',MODEL,
             '--output_dir',attempt,'--only_csv','--device', 'cuda' if device.startswith('cuda') else device], out/'inference.log', TRACK)
        shutil.copyfile(attempt/video.stem/'ball.csv',out/'ball.csv')
        # Extract display timestamps, not frame/fps estimates; supports B frames.
        with (out/'source_pts.csv').open('w') as stream:
            process = subprocess.Popen(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries',
                                        'frame=best_effort_timestamp_time','-of','csv=p=0',str(video)], stdout=subprocess.PIPE, text=True)
            for line in process.stdout:
                value = line.strip().split(',')[0]
                if value:
                    stream.write(f'{float(value):.6f}\n')
            if process.wait(): raise RuntimeError('无法提取视频显示时间戳')
        provenance.update(mode='fresh_inference', parameters={'device': device, 'confidence_threshold':.5})
    save_json(out/'provenance.json',provenance)
    return str(out/'ball.csv'), [out/'ball.csv',out/'source_pts.csv',out/'provenance.json']


def ingest_player(video, directory, number, python, device, confidence):
    out = directory/'player'; out.mkdir(exist_ok=True)
    if number is None:
        save_json(out/'index.json', {'status':'not_requested','number':None,'detections':[]})
    else:
        run([python,APP/'upstream_worker.py','player','--video',video,'--output',out,
             '--number',number,'--confidence',confidence,'--device',device],out/'inference.log')
    info = project_info(PLAYER, [PLAYER/'yolov8n.pt', PLAYER/'.cache/easyocr/model/english_g2.pth', PLAYER/'.cache/easyocr/model/craft_mlt_25k.pth'])
    info['parameters'] = {'number': number, 'sample_interval_sec': 1., 'confidence':confidence}
    save_json(out/'provenance.json', info)
    return str(out/'index.json'),[out/'index.json',out/'provenance.json']
