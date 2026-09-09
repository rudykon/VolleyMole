#!/usr/bin/env python3
"""VolleyMole: a sequential, resumable rally -> ranking -> video pipeline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from common import APP, ROOT, Stages, digest, functions_digest, identity, probe, read_json, run, save_json
from adapters import ANALYTICS, TRACK, PLAYER, discover, ingest_analytics, ingest_tracking, ingest_player
from rally import build_manifest
from ranker import rank
from schemas import validate_decision


def verify(directory, top_k):
    manifest=read_json(directory/'match_manifest.json');decision=read_json(directory/'edit_decision.json')
    validate_decision(decision,manifest,directory,top_k)
    render=read_json(directory/'render_report.json')
    if len(render['clips'])!=top_k or [c['rank'] for c in render['clips']]!=list(range(top_k,0,-1)):
        raise ValueError('成片回合数量或播放顺序错误')
    reports=[]
    for item in render['clips']+[{'path':render['output'],'duration_sec':render['expected_duration_sec']}]:
        path=Path(item['path'])
        data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
        video=next(s for s in data['streams'] if s['codec_type']=='video')
        audio=next(s for s in data['streams'] if s['codec_type']=='audio')
        if (video['width'],video['height'])!=(720,1280) or video['codec_name']!='h264':
            raise ValueError('输出不是 H.264 9:16 视频')
        expected_frames=round(item['duration_sec']*30)
        if video['r_frame_rate']!='30/1' or int(video['nb_frames'])!=expected_frames:
            raise ValueError(f'输出帧率或帧数不正确：{path.name}')
        av_delta=abs(float(video['duration'])-float(audio['duration']))
        av_start_delta=abs(float(video.get('start_time',0))-float(audio.get('start_time',0)))
        duration_error=abs(float(video['duration'])-item['duration_sec'])
        if av_delta>.12 or duration_error>.12 or av_start_delta>.08:
            raise ValueError(f'音视频时间校验失败：{path.name}')
        log=directory/'verification'/f'{path.stem}.log'
        run(['ffmpeg','-v','error','-xerror','-i',path,'-f','null','-'],log)
        reports.append({'path':str(path),'sha256':digest(path),'duration_sec':float(video['duration']),
                        'av_duration_delta_sec':av_delta,'av_start_delta_sec':av_start_delta,
                        'duration_error_sec':duration_error,'full_decode':'passed','frames':int(video['nb_frames'])})
    report={'status':'passed','top_k':top_k,'videos':reports,'source_has_audio':manifest['source']['has_audio'],
            'ranking_mode':decision['ranking_mode'],'note':'完整解码与时间轴校验不等于语义识别准确率。'}
    save_json(directory/'verification.json',report)
    return str(directory/'verification.json'),[directory/'verification.json']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--top-k',type=int,choices=(5,10),default=5)
    parser.add_argument('--focus-player',type=int)
    parser.add_argument('--format',choices=['vertical'],default='vertical')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--analytics-cache',type=Path)
    parser.add_argument('--tracking-cache',type=Path)
    parser.add_argument('--no-cache-discovery',action='store_true')
    parser.add_argument('--analytics-python',default=str(ANALYTICS/'.venv-inference/bin/python'))
    parser.add_argument('--tracking-python',default=str(TRACK/'.venv/bin/python'))
    parser.add_argument('--player-python',default=str(PLAYER/'.venv/bin/python'))
    parser.add_argument('--device',choices=['auto','cpu','cuda','cuda:0'],default='auto')
    parser.add_argument('--ranker',choices=['auto','rules'],default='auto')
    parser.add_argument('--api-base',default=os.getenv('VOLLEYMOLE_API_BASE','https://api.openai.com/v1'))
    parser.add_argument('--model',default=os.getenv('VOLLEYMOLE_MODEL'))
    parser.add_argument('--vision-model',default=os.getenv('VOLLEYMOLE_VISION_MODEL'),help='可选独立视觉评审模型；主模型不支持图像时默认从同一服务自动选择')
    parser.add_argument('--api-timeout',type=float,default=240)
    parser.add_argument('--llm-config',type=Path,default=ROOT/'llm_api.json',help='读取 llm 节点；rank 节点为文本 reranker，不用于生成剪辑单')
    parser.add_argument('--font',default='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    parser.add_argument('--stop-after',choices=['manifest','rank','render'],default='render')
    parser.add_argument('--rerun-from',choices=['analytics','tracking','player','rallies','previews','rank','render','verify'])
    args=parser.parse_args()
    if args.llm_config.is_file():
        llm=read_json(args.llm_config).get('llm',{})
        if llm.get('provider') not in (None,'openai_compatible'):
            parser.error('当前只支持 openai_compatible 格式接口')
        args.api_base=llm.get('base_url',args.api_base)
        args.model=args.model or llm.get('model')
        args.vision_model=args.vision_model or llm.get('vision_model')
        if llm.get('api_key'):
            os.environ['VOLLEYMOLE_API_KEY']=llm['api_key']
    video=args.video.resolve()
    if not video.is_file():parser.error(f'找不到素材：{video}')
    if args.focus_player is not None and not 0<=args.focus_player<=999:parser.error('球衣号码应为 0–999')
    directory=(args.output or ROOT/'runs'/f'{video.stem}-top{args.top_k}').resolve()
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('该比赛目录已由另一个进程使用')
    config=read_json(APP/'defaults.json')
    if args.config:
        override=read_json(args.config)
        unknown=set(override)-set(config)
        if unknown:parser.error(f'未知配置项：{sorted(unknown)}')
        config['weights'].update(override.pop('weights',{}));config.update(override)
    for k in ('bin_sec','bridge_gap_sec','min_rally_sec','max_rally_sec','preview_limit'):
        if not isinstance(config[k],(int,float)) or config[k]<=0:parser.error(f'{k} 必须为正数')
    meta=probe(video);meta['identity']=identity(video)
    code={str(p.relative_to(APP)):digest(p) for p in APP.glob('*.py')}
    stages=Stages(directory)
    if args.rerun_from:
        order=['analytics','tracking','player','rallies','previews','rank','render','verify']
        for name in order[order.index(args.rerun_from):]:
            stages.data['stages'].pop(name,None)
        save_json(stages.path,stages.data)
    cached={} if args.no_cache_discovery else discover(video,meta['frame_count'])
    analytics_cache=args.analytics_cache or cached.get('analytics')
    tracking_cache=args.tracking_cache or cached.get('tracking')
    save_json(directory/'run_config.json',{'video':str(video),'top_k':args.top_k,'focus_player':args.focus_player,
              'analytics_cache':str(analytics_cache) if analytics_cache else None,'tracking_cache':str(tracking_cache) if tracking_cache else None,
              'device':args.device,'config':config,'code':code})
    def cache_sig(path, files):
        return [identity(Path(path)/f) for f in files] if path else None
    analytics=stages.execute('analytics',{'source':meta['identity'],'cache':cache_sig(analytics_cache,['summary.json','detections.jsonl']),
                            'worker':code['upstream_worker.py'],'adapter':code['adapters.py'],'python':args.analytics_python,'device':args.device},
                            lambda:ingest_analytics(video,directory,analytics_cache,args.analytics_python,args.device))
    tracking=stages.execute('tracking',{'source':meta['identity'],'cache':cache_sig(tracking_cache,['review_metrics.json',f'{video.stem}/ball.csv','source_pts.csv']),
                           'adapter':code['adapters.py'],'python':args.tracking_python,'device':args.device},
                           lambda:ingest_tracking(video,directory,tracking_cache,args.tracking_python,args.device))
    player=stages.execute('player',{'source':meta['identity'],'number':args.focus_player,'threshold':config['player_confidence'],
                         'worker':code['upstream_worker.py'],'adapter':code['adapters.py'],'python':args.player_python,'device':args.device},
                         lambda:ingest_player(video,directory,args.focus_player,args.player_python,args.device,config['player_confidence']))
    def make_manifest():
        data,files=build_manifest(meta,analytics,tracking,directory/'tracking/source_pts.csv',player,directory,config,
                                 {k:read_json(directory/k/'provenance.json') for k in ('analytics','tracking','player')})
        return str(directory/'match_manifest.json'),files
    manifest_path=stages.execute('rallies',{'analytics':digest(analytics),'tracking':digest(tracking),'pts':digest(directory/'tracking/source_pts.csv'),
                                'player':digest(player),'config':config,'code':code['rally.py'],'source':meta},make_manifest)
    if args.stop_after=='manifest':return
    manifest=read_json(manifest_path)
    def make_previews():
        run([args.tracking_python,APP/'media_worker.py','previews','--run',directory],directory/'previews.log')
        index=read_json(directory/'previews/index.json')
        return str(directory/'previews/index.json'),[directory/'previews/index.json']+[directory/p for p in index['files']]
    preview_signature={'source':meta['identity'],'samples':[(r['preview_times_sec'],r['preview_frames']) for r in manifest['rallies']],
                       'code':functions_digest(APP/'media_worker.py',{'previews','frame_at'})}
    stages.execute('previews',preview_signature,make_previews)
    decision=stages.execute('rank',{'manifest':digest(manifest_path),'k':args.top_k,'focus':args.focus_player,'ranker':args.ranker,
                            'model':args.model,'endpoint':args.api_base,'has_key':bool(os.getenv('VOLLEYMOLE_API_KEY') or os.getenv('OPENAI_API_KEY')),
                            'code':code['ranker.py'],'schema':code['schemas.py'],'semantic':code['semantic.py'],'vision_model':args.vision_model,
                            'prompt':digest(APP/'prompts/rank_top_plays.md'),'vision_prompt':digest(APP/'prompts/review_frames.md')},
                            lambda:rank(manifest,directory,args.top_k,args.focus_player,args.ranker,args.api_base,args.model,args.api_timeout,args.vision_model))
    if args.stop_after=='rank':return
    def make_video():
        run([args.tracking_python,APP/'media_worker.py','render','--run',directory,'--font',args.font],directory/'render.log')
        report=read_json(directory/'render_report.json')
        return report['output'],[report['output'],directory/'render_report.json']+[c['path'] for c in report['clips']]
    output=stages.execute('render',{'decision':digest(decision),'manifest':digest(manifest_path),'code':code['media_worker.py'],
                          'schema':code['schemas.py'],'font':identity(args.font)},make_video)
    stages.execute('verify',{'output':digest(output),'decision':digest(decision),'report':digest(directory/'render_report.json'),
                             'code':code['run_match.py']},lambda:verify(directory,args.top_k))
    print(f'VolleyMole 成片：{output}',flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(f'VolleyMole: {exc}',file=sys.stderr);sys.exit(1)
