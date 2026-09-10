"""Opt-in actual OCR model test with synthetic, explicitly known digits (not match accuracy)."""
import argparse
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from volleymole.common import save_json
from volleymole.models import ModelRegistry
from volleymole.video import FramePacket
from volleymole.jersey import JerseyReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (340,600), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((100,190),'12',fill='black',font=ImageFont.truetype('DejaVuSans-Bold.ttf',100))
    image.save(args.output/'known-number.png')
    reader = JerseyReader(ModelRegistry(args.models),12,'cpu',runtime_directory=args.output/'ocr_runtime')
    pixels = np.asarray(image)[:,:,::-1].copy()
    for index in (0,30):
        reader.consume(FramePacket(index,index,Fraction(1,30),0.,pixels),
                       [{'xyxy':[0,0,340,600],'confidence':1.}])
    result = reader.result()
    result['fixture'] = 'synthetic known torso rectangle and digits; no person model; not real-match jersey accuracy'
    result['test_status'] = 'passed' if {r['time_sec'] for r in result['detections']}=={0.,1.} else 'failed'
    save_json(args.output/'result.json',result)
    print(result['test_status'], 'accepted samples:', len(result['detections']))
    if result['test_status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
