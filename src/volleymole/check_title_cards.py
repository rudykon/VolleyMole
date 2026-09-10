"""Verify readable title holds in encoded interludes AND the final compilation."""
import argparse
from pathlib import Path
import sys
import subprocess
import cv2
import numpy as np

from .common import read_json,save_json
from .illustrated import headers,rank_label


def frame_at(video,seconds,width=720):
    # UI checks need lossless decoding: a JPEG screenshot adds ringing around
    # outlined glyphs and can falsely fail a correctly encoded transparent title.
    raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(seconds),'-i',str(video),'-frames:v','1',
                                 '-vf',f'scale={width}:-2','-f','image2pipe','-vcodec','png','pipe:1'])
    pixels=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if pixels is None:raise ValueError('无法无损解码标题检查帧')
    return pixels


def check(directory):
    report=read_json(directory/'render_report_lively.json')
    decision=read_json(directory/'edit_decision.json')
    source=read_json(directory/'match_manifest.json')['source']
    by_rank={item['rank']:item for item in decision['selected']}
    top_k=len(by_rank)
    results=[]
    for segment in report['segments']:
        if segment['kind']!='transition':continue
        label=rank_label(segment['next_rank'],top_k)
        if segment['rank_label']!=label:raise ValueError('转场名次标识不正确')
        card_path=Path(segment['path']).with_suffix('.png')
        card=cv2.imread(str(card_path))
        if card is None:raise ValueError('缺失实际插画标题卡')
        samples=[]
        for relative in (.5,1.5,2.5):
            individual=frame_at(segment['path'],relative,720)
            combined=frame_at(report['output'],segment['timeline_start_sec']+relative,720)
            errors={name:float(np.mean(abs(pixels.astype(float)-card.astype(float))))
                    for name,pixels in [('individual',individual),('compilation',combined)]}
            if max(errors.values())>5:
                raise ValueError(f'转场标题没有稳定完整展示：{segment["index"]}, {errors}')
            samples.append({'relative_sec':relative,'pixel_mean_absolute_error':errors})
        results.append({'rank':segment['next_rank'],'rank_label':label,'display_title':segment['display_title'],
                        'duration_sec':segment['duration_sec'],'title_hold_sec':segment['title_hold_sec'],
                        'samples':samples})
    if len(results)!=len(report['clips']):raise ValueError('可读标题卡数量错误')
    replay_overlay_checks=[]
    if report.get('replay_caption_background')=='transparent':
        for segment in report['segments']:
            if segment['kind']!='replay':continue
            overlay=cv2.imread(str(Path(segment['path']).with_suffix('.png')),cv2.IMREAD_UNCHANGED)
            if overlay is None or overlay.shape!=(1280,720,4):raise ValueError('缺失透明回放提示叠加层')
            alpha=overlay[:,:,3]
            clear_ratio=float(np.mean(alpha[124:194,22:387]==0))
            if clear_ratio<=.6 or any(alpha[y,x]!=0 for x,y in ((370,160),(200,190),(50,190))):
                raise ValueError('回放提示仍包含非透明底板')
            replay_overlay_checks.append({'rank':segment['rank'],'transparent_pixel_ratio':clear_ratio,'status':'passed'})
    # Compare the rank glyphs themselves, not just a mostly unchanged header.
    # Luma avoids falsely rejecting yuv420p chroma subsampling around colorful ink.
    candidate_layers={rank:headers({'rank':rank,'title':'排球'},report['font_path'],top_k,'排球')[-1]
                      for rank in by_rank}
    candidates={rank:cv2.cvtColor(layer,cv2.COLOR_BGRA2GRAY)[30:80,50:260].astype(float)
                for rank,layer in candidate_layers.items()}
    rank_mask=np.logical_and.reduce([layer[30:80,50:260,3]>=250 for layer in candidate_layers.values()])
    rank_mask=cv2.erode(rank_mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    header_results=[]
    live_background_results=[]
    for clip in report['clips']:
        item=by_rank[clip['rank']]
        expected=headers(item,report['font_path'],top_k,clip['display_title'])[-1]
        if expected.shape!=(110,720,4):raise ValueError('标题层必须保留透明通道')
        opaque=cv2.erode((expected[:,:,3]>=250).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        clear=cv2.erode((expected[:,:,3]==0).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        if float(np.mean(clear))<.35:raise ValueError('顶部标题层仍有大面积底板')
        label=rank_label(clip['rank'],top_k)
        if clip['rank_label']!=label:raise ValueError('回合顶部名次标识不正确')
        for segment in report['segments']:
            if segment['kind'] not in ('rally','replay') or segment['rank']!=clip['rank']:continue
            # Past the 0.4 s artwork entrance, while the rank itself stays fixed.
            relative=min(.75,segment['duration_sec']/2)
            pixels=frame_at(report['output'],segment['timeline_start_sec']+relative,720)[:110]
            error=float(np.mean(abs(pixels.astype(float)-expected[:,:,:3].astype(float))[opaque]))
            gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY).astype(float)
            luma_error=float(np.mean(abs(gray-cv2.cvtColor(expected,cv2.COLOR_BGRA2GRAY).astype(float))[opaque]))
            scores=sorted((float(np.mean(((gray[30:80,50:260]-template)**2)[rank_mask])),rank)
                          for rank,template in candidates.items())
            if luma_error>4 or scores[0][1]!=clip['rank'] or scores[0][0]>150 or scores[0][0]>.5*scores[1][0]:
                raise ValueError(f'成片回合或回放顶部名次不完整或不匹配：{label}, {luma_error}, {scores[:2]}')
            base_relative=segment['source_start_sec']-clip['source_start_sec']+relative*segment['playback_rate']
            base=frame_at(clip['path'],base_relative,720)[:110]
            background_error=float(np.mean(abs(pixels.astype(float)-base.astype(float))[clear]))
            if background_error>12:raise ValueError('回合或回放顶部未保留源回合画面')
            header_results.append({'kind':segment['kind'],'rank_label':label,'opaque_pixel_mean_absolute_error':error,
                                   'luma_mean_absolute_error':luma_error,'recognized_rank':scores[0][1],
                                   'rank_template_mse':scores[0][0],'nearest_other_rank_mse':scores[1][0],
                                   'transparent_pixel_ratio':float(np.mean(clear)),'background_match_error':background_error})
        # Reconstruct the camera crop from original footage, independently of the
        # encoded header: transparent must mean live source pixels, not blank black.
        layout=clip['detail_layout']
        if (layout['top'],layout['height'],layout['overview_top'])!=(0,875,875):
            raise ValueError('比赛画面没有延伸到顶部或全场小画面位置发生变化')
        if not 1<=len(clip['header_source_samples'])<=3:raise ValueError('缺少顶部源画面对照采样')
        for sample in clip['header_source_samples']:
            original=frame_at(source['path'],sample['source_sec'],source['width'])
            left=sample['crop_left'];width=layout['source_crop_width']
            background=cv2.resize(original[:layout['source_height'],left:left+width],(720,875),
                                  interpolation=cv2.INTER_AREA)[:110]
            relative=sample['output_frame']/30
            errors={}
            for name,path,when in (('individual',clip['path'],relative),
                                   ('compilation',report['output'],clip['timeline_start_sec']+relative)):
                actual=frame_at(path,when,720)[:110]
                errors[name]=float(np.mean(abs(actual.astype(float)-background.astype(float))[clear]))
            if max(errors.values())>12:raise ValueError(f'顶部不是对应的比赛源画面：{label}, {errors}')
            live_background_results.append({'rank':clip['rank'],'source_sec':sample['source_sec'],
                                            'transparent_region_mean_absolute_error':errors})
    save_json(directory/'title_card_readability.json',{'status':'passed',
        'method':'Compare cards, opaque title glyphs, rank templates, and transparent title regions against original source camera crops',
        'results':results,'header_results':header_results,'replay_overlay_checks':replay_overlay_checks,
        'live_background_results':live_background_results})
    print('转场、名次和回放提示检查通过；正常回合与回放顶部显示真实比赛画面，无黑底标题栏')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    check(parser.parse_args().run.resolve())
