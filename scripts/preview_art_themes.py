#!/usr/bin/env python3
"""Render the existing video UI with built-in artwork; never edit source PNGs."""
import argparse
from html import escape
from pathlib import Path

from volleymole.art_themes import THEME_IDS, get_theme
from volleymole.common import DEFAULT_FONT, save_json
from volleymole.illustrated import ART_NAMES, headers, overlay, title_card
from volleymole.presentation import lively_title


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/art-themes'))
    parser.add_argument('--themes', nargs='+', choices=THEME_IDS, default=list(THEME_IDS[1:]))
    parser.add_argument('--font', default=str(DEFAULT_FONT))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    # A fresh directory prevents accidentally replacing a previous visual audit.
    if any((args.output / name).exists() for name in args.themes):
        parser.error('预览目录已包含所选风格，请指定新的 --output 目录')
    from PIL import Image
    import cv2
    import numpy as np
    sections, results = [], []
    actions = ('排球好球', '低姿救球', '二传组织配合', '极低姿态接球', '低位防守', '长回合起跳对抗')
    for name in args.themes:
        theme = get_theme(name)
        destination = args.output / name
        destination.mkdir()
        assets = []
        for asset in ART_NAMES:
            with Image.open(theme.root / f'{asset}.png') as image:
                if image.mode != 'RGBA' or image.getchannel('A').getextrema() != (0, 255):
                    raise ValueError(f'素材未保留完整透明通道：{name}/{asset}')
                assets.append({'asset':asset, 'size':list(image.size), 'mode':image.mode})
        cards = []
        for index, title in enumerate(actions):
            top_k = 5 if index < 5 else 10
            rank = index + 1 if index < 5 else 10
            item = {'rank':rank, 'title':title}
            filename = f'card-{index + 1}.png'
            title_card(item, lively_title(item), args.font, top_k, name).save(destination / filename)
            cards.append(f'<img loading="lazy" src="{name}/{filename}" alt="{escape(title)}">')
        for top_k in (5, 10):
            for rank in range(1, top_k + 1):
                item = {'rank':rank, 'title':'排球好球'}
                layer = headers(item, args.font, top_k, lively_title(item), name)[-1]
                Image.fromarray(cv2.cvtColor(layer, cv2.COLOR_BGRA2RGBA)).save(destination / f'header-{top_k}-{rank}.png')
                if np.mean(layer[:, :, 3] == 0) <= .45:
                    raise ValueError(f'顶部透明区域不足：{name} / {top_k} / {rank}')
        for index in range(5):
            overlay(destination / f'teaser-{index}.png', 'teaser', args.font, index, name)
        overlay(destination / 'replay.png', 'replay', args.font, art_theme=name)
        sections.append(f'<section><h2>{escape(theme.label)} <code>--art-theme {name}</code></h2><div>{"".join(cards)}</div></section>')
        results.append({'theme':name, 'assets':assets, 'cards':6, 'rank_headers':15, 'teasers':5, 'replay_overlays':1})
        print(f'{theme.label}: 7 张源素材、6 张标题卡、15 个名次、5 个片头角标、1 个回放提示检查完成', flush=True)
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VolleyMole 插画风格预览</title><style>body{margin:32px;font-family:system-ui;background:#17212b;color:#f7efd9}h1{font-size:32px}code{font-size:14px;color:#9fd8c2}section{margin:40px 0}section div{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}img{width:100%;border-radius:12px}p{line-height:1.8;color:#c0cbd4}</style><h1>VolleyMole · 五套插画风格</h1><p>实际视频标题卡版式预览，不是比赛检测结果。原始生成素材保持不变；中文名次由程序排版。各风格目录另含全部五佳／十佳排名角标及片头、回放透明层。</p>' + ''.join(sections) + '</html>'
    (args.output / 'index.html').write_text(document, encoding='utf-8')
    save_json(args.output / 'validation.json', {'status':'passed', 'themes':results})
    print(f'打开预览：{args.output / "index.html"}')


if __name__ == '__main__':
    main()
