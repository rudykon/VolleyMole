"""Render the real layout at native size for visual review before video encoding."""
import argparse
from pathlib import Path
import sys

import cv2
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import read_json
from illustrated import headers,prepare_brand,title_card
from presentation import lively_title


def preview(directory,output):
    output.mkdir(parents=True,exist_ok=True)
    prepare_brand()
    decision=read_json(directory/'edit_decision.json')
    items=list(reversed(decision['selected']))
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    sheet=Image.new('RGB',(360*len(items),695),'white')
    for i,item in enumerate(items):
        title=lively_title(item)
        card=title_card(item,title,font,len(items))
        card.save(output/f'card_{item["rank"]:02d}.png')
        header=Image.fromarray(cv2.cvtColor(headers(item,font,len(items),title)[-1],cv2.COLOR_BGRA2RGBA))
        header.save(output/f'header_{item["rank"]:02d}.png')
        sheet.paste(card.resize((360,640),Image.Resampling.LANCZOS),(i*360,0))
        reduced=header.resize((360,55),Image.Resampling.LANCZOS)
        sheet.paste(reduced,(i*360,640),reduced)
    sheet.save(output/'design_overview.png')
    # The widest supported ordinal must also fit the exact same layout.
    item={'rank':10,'title':'排球好球'}
    title_card(item,lively_title(item),font,10).save(output/'card_top10_tenth.png')
    Image.fromarray(cv2.cvtColor(headers(item,font,10,lively_title(item))[-1],cv2.COLOR_BGRA2RGBA)).save(output/'header_top10_tenth.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    preview(args.run,args.output)
