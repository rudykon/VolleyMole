"""Render a bilingual typography gallery using the real video renderer."""
import argparse
from html import escape
from pathlib import Path
import cv2
from PIL import Image
from volleymole.art_themes import THEME_IDS, get_theme
from volleymole.common import DEFAULT_FONT, save_json
from volleymole.illustrated import headers, overlay, title_card
from volleymole.presentation import lively_title
from volleymole.title_templates import STYLES, LABELS


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/title-templates'))
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    sections=[];records=[]
    item={'rank':10,'title':'长回合起跳对抗与防守站位'}
    for style in STYLES:
        columns=[]
        for lang in ('zh','en'):
            name=f'{style}-{lang}';directory=args.output/name;directory.mkdir()
            title=lively_title(item,name)
            for art in THEME_IDS[1:]:
                title_card(item,title,DEFAULT_FONT,10,art,name).save(directory/f'{art}.png')
            # A second action and a long headline exercise different line lengths.
            other={'rank':1,'title':'持续攻防与二传组织配合'}
            title_card(other,lively_title(other,name),DEFAULT_FONT,5,'papercut',name).save(directory/'setting.png')
            for k in (5,10):
                for rank in range(1,k+1):
                    layer=headers({'rank':rank,'title':item['title']},DEFAULT_FONT,k,title,'manga',name)[-1]
                    Image.fromarray(cv2.cvtColor(layer,cv2.COLOR_BGRA2RGBA)).save(directory/f'header-{k}-{rank}.png')
            overlay(directory/'teaser.png','teaser',DEFAULT_FONT,0,'manga',name)
            overlay(directory/'replay.png','replay',DEFAULT_FONT,0,'manga',name)
            columns.append(f'<article><h3>{"中文版" if lang=="zh" else "English"}</h3><code>--title-template {name}</code><img class="card" data-template="{name}" src="{name}/manga.png" alt="{escape(title)}"><img class="header" src="{name}/header-10-10.png" alt="Rank 10 header"></article>')
            records.append({'template':name,'art_themes':list(THEME_IDS[1:]),'rank_headers':15})
            print(f'{name}: 6 cards, 15 rank headers, teaser and replay rendered',flush=True)
        sections.append(f'<section><h2>{LABELS[style]} <small>{style.upper()}</small></h2><div class="pair">{"".join(columns)}</div></section>')
    options=''.join(f'<option value="{name}">{get_theme(name).label}</option>' for name in THEME_IDS[1:])
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VolleyMole — 中英标题设计</title><style>body{margin:0;background:#121a20;color:#f2f0e9;font-family:system-ui}main{max-width:1000px;margin:auto;padding:48px 24px}h1{font-size:38px;letter-spacing:-1px}p{color:#b5c1c9;line-height:1.8}.tools{position:sticky;top:0;padding:16px;background:#121a20ed;z-index:2;border-bottom:1px solid #45515a}select{padding:10px;border-radius:6px;background:#eee9db;color:#17212b}section{margin-top:64px}h2{font-size:28px}small{font-size:12px;color:#89bbaa;letter-spacing:3px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{min-width:0}h3{font-weight:500}code{display:block;color:#94c9b6;font-size:12px;margin-bottom:16px}.card{width:100%;border-radius:8px}.header{width:100%;margin-top:16px;background:#53636d;border-radius:6px}@media(max-width:620px){.pair{grid-template-columns:1fr}}</style><main><h1>标题，不止是换一种字体。</h1><p>五组设计 × 中英双语。真实 720 × 1280 视频版式；英文独立字体、文案与断行。下方小图是实际视频角标。十套模板可与五套插画自由组合。</p><div class="tools">插画与配色 <select id="art">'''+options+'</select></div>'+''.join(sections)+'''</main><script>document.querySelector('#art').addEventListener('change',event=>{document.querySelectorAll('.card').forEach(img=>{img.src=img.dataset.template+'/'+event.target.value+'.png'})})</script></html>'''
    (args.output/'index.html').write_text(html,encoding='utf-8')
    save_json(args.output/'manifest.json',{'templates':records,'note':'Code-native title layout previews, not generated match evidence.'})
    print(args.output/'index.html')


if __name__=='__main__':main()
