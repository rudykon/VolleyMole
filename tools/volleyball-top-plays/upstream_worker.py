"""Thin adapters importing upstream code, executed only in the respective venv."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from common import ROOT, save_json

PLAYER_SAMPLE_FILTER = 'fps=1:round=up:start_time=0'


def analytics(args):
    import av
    import cv2
    import torch
    root = ROOT/'tools/volleyball_analytics'
    sys.path.insert(0,str(root/'src'))
    os.chdir(root)
    from run_sample import detection_json
    from run_batch_video import detect_batch
    from ml_manager.ml_manager import MLManager
    from ml_manager.settings import ModelWeightsConfig
    torch.set_num_threads(4)
    device = 'cuda:0' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    manager = MLManager(ModelWeightsConfig(player_detection=str(root/'weights/yolo11l-pose.pt')), device=device)
    manager.check_models()
    if not manager.is_model_available('game_state_classification'):
        raise RuntimeError('比赛状态模型未正确加载')
    container = av.open(str(args.video))
    video_stream = container.streams.video[0]
    from common import probe
    rotation = probe(args.video)['rotation']
    count, first, begin = 0, None, time.monotonic()
    windows = []
    iterator = iter(container.decode(video_stream))
    with (args.output/'detections.jsonl').open('w') as out:
        while True:
            chunk = []
            for _ in range(30):
                try: frame = next(iterator)
                except StopIteration: break
                if frame.pts is None: raise ValueError('视频帧缺少 PTS')
                pts = float(frame.pts*frame.time_base)
                if first is None: first = pts
                pixels = frame.to_ndarray(format='bgr24')
                if rotation:
                    pixels = cv2.rotate(pixels,{90:cv2.ROTATE_90_COUNTERCLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_CLOCKWISE}[rotation])
                chunk.append((pixels,pts))
            if not chunk: break
            images = [f[0] for f in chunk]
            state_images = [cv2.resize(f,(224,224)) for f in images]
            state_images += [state_images[-1]]*max(0,16-len(state_images))
            state = manager.classify_game_state(state_images)
            if str(state.predicted_class) == 'unknown': raise RuntimeError('比赛状态推理失败')
            detections = detect_batch(manager,images,8)
            windows.append({'start_s':chunk[0][1]-first,'end_s':chunk[-1][1]-first,'state':str(state.predicted_class)})
            for (_,pts),(actions,ball,players) in zip(chunk,detections):
                row={'frame':count,'source_time_s':pts,'time_s':pts-first,'state':str(state.predicted_class),'state_confidence':float(state.confidence),
                     'ball':detection_json(ball) if ball else None, 'actions':[detection_json(a) for a in actions],
                     'players':[detection_json(p) for p in players if p.bbox is not None]}
                out.write(json.dumps(row)+'\n'); count+=1
            if count%900==0: print(f'analytics: {count} frames',flush=True)
    container.close()
    if not count: raise ValueError('输入没有可解码的视频帧')
    save_json(args.output/'summary.json',{'status':'complete','input':str(args.video),'processed_frames':count,
              'source_start_s':first,'state_windows':windows,'device':device,'elapsed_sec':time.monotonic()-begin})


def player(args):
    root = ROOT/'tools/volleyball-highlights'
    os.environ.setdefault('EASYOCR_MODULE_PATH',str(root/'.cache/easyocr'))
    os.environ.setdefault('YOLO_CONFIG_DIR',str(args.output/'yolo_config'))
    sys.path.insert(0,str(root)); os.chdir(root)
    import cv2
    from volleyball_highlights import PlayerDetector
    detector = PlayerDetector(args.number,args.confidence,args.device.startswith('cuda'))
    # round=up samples at each integer second, whereas default round=near
    # would label a frame from t=0.467 as t=0 on a 30-fps source.
    import subprocess
    from common import probe
    meta = probe(args.video); w,h=meta['width'],meta['height']
    import numpy as np
    process = subprocess.Popen(['ffmpeg','-v','error','-i',str(args.video),'-vf',PLAYER_SAMPLE_FILTER,'-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE)
    detections=[]; i=0
    try:
        while True:
            raw=process.stdout.read(w*h*3)
            if not raw: break
            if len(raw)!=w*h*3: raise ValueError('球员识别解码帧不完整')
            for confidence,bbox in detector.detect(np.frombuffer(raw,np.uint8).reshape(h,w,3)):
                detections.append({'time_sec':float(i),'number':args.number,'confidence':float(confidence),'bbox':list(bbox)})
            i+=1
        if process.wait(): raise RuntimeError('球员识别视频解码失败')
    finally:
        if process.poll() is None: process.terminate(); process.wait()
    save_json(args.output/'index.json',{'status':'complete','number':args.number,'sample_interval_sec':1.,'sample_count':i,'detections':detections,
              'note':'号码仅作为出现证据；至少两个高置信度样本才增加回合权重。无可靠命中时输出通用集锦。'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['analytics','player'])
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--number',type=int)
    parser.add_argument('--confidence',type=float,default=.75)
    args=parser.parse_args(); args.video=args.video.resolve(); args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    {'analytics':analytics,'player':player}[args.kind](args)
