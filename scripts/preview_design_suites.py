"""Five complete identities, bilingual galleries and real presentation samples."""
import argparse
from html import escape
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from volleymole.common import DEFAULT_FONT, digest, read_json, save_json, probe
from volleymole.design_suites import SUITES, SuiteCard, design_manifest, suite_headers, suite_overlay
from volleymole.illustrated import encode_transition
from volleymole.presentation import effect_clip, lively_title
from volleymole.media_worker import render_clip, concatenate_segments
from volleymole.check_title_cards import frame_at, normalized_header
from volleymole.quality import QUALITY_IDS,DEFAULT_QUALITY,get_quality
from preview_transitions import encode_frames, decode


def select_gallery_language(page,language):
    """A single-language gallery must never hide its only set of videos."""
    if language=='both':return page
    toolbar='<div class="toolbar"><button class="active" data-lang="zh">中文</button><button data-lang="en">English</button></div>'
    label='English' if language=='en' else '中文'
    page=page.replace(toolbar,f'<div class="toolbar"><button class="active" data-lang="{language}">{label}</button></div>')
    page=page.replace(f'<article data-language="{language}" hidden>',f'<article data-language="{language}">')
    if language=='en':
        page=page.replace('不是五种装饰。<br>是五套完整的视觉语言。','Five styles.<br>One complete English collection.')
        page=page.replace('每套包含中文和英文版本。','本页展示五套英文版本，含英文标题、名次、角标与回放提示。')
        for suite in SUITES:page=page.replace(f'<h2>{escape(suite.label)}</h2>',f'<h2>{escape(suite.collection)}</h2>')
    return page


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/design-suites'))
    parser.add_argument('--language',choices=('both','zh','en'),default='both',help='预览语言：双语、仅中文或仅英文')
    parser.add_argument('--quality',choices=QUALITY_IDS,default=DEFAULT_QUALITY,help='样片画质，默认 1080p')
    parser.add_argument('--reference-run',type=Path,help='可选：已有完整运行目录，复用其证据制作九秒样片')
    parser.add_argument('--source',type=Path,help='原片移动后的路径；必须与记录 SHA-256 一致')
    args=parser.parse_args(argv)
    q=get_quality(args.quality)
    languages=('zh','en') if args.language=='both' else (args.language,)
    if args.source and not args.reference_run:parser.error('--source 需要 --reference-run')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=decision=item=rally=None
    if args.reference_run:
        manifest=read_json(args.reference_run/'match_manifest.json')
        decision=read_json(args.reference_run/'edit_decision.json')
        from volleymole.schemas import validate_decision
        validate_decision(decision,manifest,args.reference_run,len(decision['selected']))
        if args.source:
            recorded=manifest['source']['identity']
            if args.source.stat().st_size!=recorded['bytes'] or digest(args.source)!=recorded['sha256']:
                raise ValueError('原片与参考证据不一致')
            manifest['source']['path']=str(args.source.resolve())
        item=dict(decision['selected'][0])
        item['clip_end_sec']=min(item['clip_end_sec'],item['clip_start_sec']+2)
        if item['clip_end_sec']-item['clip_start_sec']<1:raise ValueError('预览至少需要一秒源素材')
        rally=next(r for r in manifest['rallies'] if r['rally_id']==item['rally_id'])
    rows=[];sections=[]
    for s in SUITES:
        cards=[]
        for language in languages:
            started=time.perf_counter()
            name=f'{s.name}-{language}';out=args.output/name;out.mkdir()
            template=s.template(language)
            display_item=item or {'rank':1,'title':'长回合起跳对抗与防守站位'}
            title=lively_title(display_item,template)
            scene=SuiteCard(display_item,title,5 if not decision else len(decision['selected']),s.name,language)
            scene.image().save(out/'card.png')
            alternate={'rank':3,'title':'极低姿态接球防守'}
            SuiteCard(alternate,lively_title(alternate,template),5,s.name,language).image().save(out/'alternate.png')
            for kind in ('teaser','replay'):suite_overlay(out/f'{kind}.png',kind,0,s.name,language)
            header=suite_headers(display_item,5 if not decision else len(decision['selected']),title,s.name,language)[-1]
            cv2.imwrite(str(out/'header.png'),header)
            if manifest:
                save_json(out/'edit_decision.json',decision)
                track=out/rally['tracking_json'];track.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(args.reference_run/rally['tracking_json'],track)
                clip=render_clip(out,display_item,manifest,DEFAULT_FONT,style='lively',art_theme=s.art,title_template=template,design_suite=s.name,quality=q.name)
                seconds=clip['duration_sec']
                teaser={'kind':'teaser','playback_rate':1.,'duration_sec':1.,'source_start_sec':item['clip_start_sec'],
                        'source_end_sec':item['clip_start_sec']+1,'output_frames':30,'path':str(out/'teaser.mp4')}
                replay={'kind':'replay','playback_rate':2/3,'duration_sec':seconds*1.5,'source_start_sec':item['clip_start_sec'],
                        'source_end_sec':item['clip_end_sec'],'output_frames':round(seconds*45),'path':str(out/'replay.mp4')}
                for segment in (teaser,replay):
                    effect_clip(clip['path'],Path(segment['path']),segment,0,out/f'{segment["kind"]}.png',s.name,q.name)
                before,after=teaser,clip
            else:
                boundaries=[]
                for rank,part in ((2,'before'),(1,'after')):
                    boundary_item=dict(display_item,rank=rank)
                    frame=SuiteCard(boundary_item,lively_title(boundary_item,template),5,s.name,language).frame()
                    path=out/f'{part}.mp4';encode_frames(path,(frame for _ in range(30)),30,quality=q.name)
                    boundaries.append({'path':str(path),'duration_sec':1.,'output_frames':30})
                before,after=boundaries
            transition={'duration_sec':3.,'output_frames':90,'next_rank':display_item['rank'],'original_title':display_item['title'],
                        'display_title':title,'top_k':5 if not decision else len(decision['selected']),'path':str(out/'transition.mp4')}
            encode_transition(before,after,out/'transition.mp4',transition,DEFAULT_FONT,s.art,template,s.transition,s.name,q.name)
            segments=[before,transition,after]+([replay] if manifest else [])
            concatenate_segments(segments,out/'sample.mp4',q.name)
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out/'sample.mp4'),'-f','null','-'],check=True)
            actual=decode(out/'transition.mp4',normalize=True)
            if len(actual)!=90:raise AssertionError('转场不是 90 帧')
            maes=[];compiled_maes=[]
            for index in (15,45,75):
                expected=scene.frame(index)
                maes.append(float(np.abs(actual[index].astype(float)-expected.astype(float)).mean()))
                compiled=frame_at(out/'sample.mp4',before['duration_sec']+index/30,720)
                compiled_maes.append(float(np.abs(compiled.astype(float)-expected.astype(float)).mean()))
            if max(maes+compiled_maes)>=5:raise AssertionError((name,maes,compiled_maes))
            frames=[cv2.resize(actual[i],(180,320),interpolation=cv2.INTER_AREA) for i in (0,3,6,9,15,45,75,78,81,84,87,89)]
            cv2.imwrite(str(out/'timeline.jpg'),np.vstack((np.hstack(frames[:6]),np.hstack(frames[6:]))))
            header_errors=[]
            if manifest:
                header=normalized_header(header,q.width)
                opaque=cv2.erode((header[:,:,3]>=250).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                header_times=(4+min(.75,seconds/2),4+seconds+min(.75,replay['duration_sec']/2))
                for when in header_times:
                    pixels=frame_at(out/'sample.mp4',when,720)[:110]
                    luma=np.abs(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY).astype(float)-cv2.cvtColor(header,cv2.COLOR_BGRA2GRAY).astype(float))
                    header_errors.append(float(luma[opaque].mean()))
                if max(header_errors)>=4:raise AssertionError((name,header_errors))
                cv2.imwrite(str(out/'live.jpg'),frame_at(out/'sample.mp4',header_times[0],720))
                cv2.imwrite(str(out/'replay.jpg'),frame_at(out/'sample.mp4',header_times[1],720))
            meta=probe(out/'sample.mp4');duration=meta['duration_sec']
            if (meta['width'],meta['height'])!=(q.width,q.height):raise AssertionError('样片分辨率错误')
            if abs(duration-sum(p['duration_sec'] for p in segments))>.12:raise AssertionError('样片时长错误')
            row={'suite':s.name,'language':language,'output_quality':q.report(),'design':design_manifest(s.name,language),'duration_sec':duration,
                 'animated_card_mae':maes,'compiled_animated_card_mae':compiled_maes,'live_and_replay_header_luma_mae':header_errors,
                 'full_decode':'passed','elapsed_sec':round(time.perf_counter()-started,3)}
            rows.append(row)
            save_json(args.output/'validation.json',{'status':'in_progress','scope':'Presentation snippets, not full-match inference or ranking validation.','samples':rows})
            cards.append(f'<article data-language="{language}"'+(' hidden' if language=='en' else '')+f'><video controls loop playsinline preload="metadata" poster="{name}/card.png" src="{name}/sample.mp4"></video><nav><a href="{name}/sample.mp4">下载样片</a><a href="{name}/card.png">标题卡</a><a href="{name}/alternate.png">另一动作版式</a><a href="{name}/timeline.jpg">动画时间线</a></nav><img class="header" src="{name}/header.png" alt="实际比赛角标"></article>')
            print(row,flush=True)
        swatches=''.join(f'<i style="background:rgb{color}"></i>' for color in (s.ink,s.paper,s.accent,s.secondary))
        sections.append(f'<section><div class="info"><span class="index">0{len(sections)+1} / {escape(s.collection)}</span><h2>{escape(s.label)}</h2><div class="swatches">{swatches}</div><code>--design-suite {s.name}</code><p>{escape({"matchday":"斜向构图、赤红强调、漫画动势。短促归位，干净揭幕。","atelier":"纸张肌理、陶土与鼠尾草色。编辑式网格，轻柔错层。","sumi":"宋体、宣纸、水墨与朱砂。留白和慢呼吸，让画面安静下来。","aurora":"深蓝空间、银白玻璃、冰蓝与冷紫。独立悬浮，通透折光。","archive":"暖纸、复古丝网、棕色画框。衬线标题与缓慢推近，衔接琥珀光泄。"}[s.name])}</p></div><div class="visual">'+''.join(cards)+'</div></section>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VolleyMole / 五套完整视觉方案</title><style>*{box-sizing:border-box}body{margin:0;background:#101719;color:#eeeae0;font-family:system-ui}main{max-width:1120px;margin:auto;padding:64px 32px}h1{font-size:clamp(34px,6vw,64px);line-height:1.15;letter-spacing:-2px;font-weight:650;max-width:860px}header p{color:#aab7b5;line-height:1.9;max-width:720px}.eyebrow,.index{font-size:12px;letter-spacing:2px;color:#9eb5ad}.toolbar{position:sticky;top:0;z-index:3;padding:16px 0;background:#101719ee;border-bottom:1px solid #35463e}button{border:1px solid #5a6b62;border-radius:4px;background:transparent;color:#eeeae0;padding:10px 22px;margin-right:8px;cursor:pointer}button.active{background:#e9e5d9;color:#112018}section{display:grid;grid-template-columns:1fr 440px;gap:64px;padding:64px 0;border-bottom:1px solid #35463e}.info{padding-top:30px}h2{font-size:38px;font-weight:550;margin:22px 0}.swatches{display:flex;gap:8px;margin:26px 0}.swatches i{width:34px;height:34px;border-radius:50%;border:1px solid #ffffff30}code{color:#a9cfba}.info p{color:#b3c0b9;line-height:1.9;max-width:380px;margin-top:32px}video{width:100%;display:block;border-radius:9px;background:#080c0b}nav{display:flex;gap:14px;flex-wrap:wrap;line-height:2;margin:16px 0}a{font-size:13px;color:#bddbc8}.header{width:100%;background:#526264;border-radius:5px}footer{color:#9aac9f;font-size:13px;line-height:1.9;margin-top:48px}[hidden]{display:none!important}@media(max-width:800px){section{grid-template-columns:1fr;gap:24px;padding:36px 0}.visual{max-width:440px;width:100%;margin:auto}main{padding:32px 22px}.info p{max-width:none}}</style><main><header><div class="eyebrow">VOLLEYMOLE / ART-DIRECTED COLLECTION</div><h1>不是五种装饰。<br>是五套完整的视觉语言。</h1><p>标题、插图、配色、转场、角标与回放共同设计。每套包含中文和英文版本。品牌标志保留原样；比赛画面保留原貌；运动只服务于节奏和阅读。</p></header><div class="toolbar"><button class="active" data-lang="zh">中文</button><button data-lang="en">English</button></div>'''+''.join(sections)+'''<footer>五套设计均可用一个参数启用；英文增加 --design-language en。此页是呈现层样片，不是重新推理或整场排名验收。另一动作版式是独立装饰性排版示例，不代表当前录像中的动作。玻璃排球由 Codex 内置 image_gen 新生成；其余原始插图保留，排版与动画由程序实现。</footer></main><script>document.querySelectorAll('button').forEach(button=>button.onclick=()=>{document.querySelectorAll('video').forEach(video=>video.pause());document.querySelectorAll('[data-language]').forEach(article=>article.hidden=article.dataset.language!==button.dataset.lang);document.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b===button))});</script></html>'''
    html=select_gallery_language(html,args.language)
    (args.output/'index.html').write_text(html,encoding='utf-8')
    save_json(args.output/'validation.json',{'status':'passed','scope':'Presentation snippets, not full-match inference or ranking validation.','samples':rows})
    print(args.output/'index.html',flush=True)


if __name__=='__main__':main()
