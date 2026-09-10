#!/usr/bin/env python3
"""VolleyMole: a sequential, resumable rally -> ranking -> video pipeline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from .common import APP, ROOT, DEFAULT_FONT, Stages, digest, functions_digest, identity, probe, read_json, run, save_json
from .adapters import ingest_analytics, ingest_tracking, ingest_player, ingest_shared, inference_signature
from .models import ModelRegistry
from .rally import build_manifest
from .ranker import rank, shortlist, rule_decision
from .schemas import validate_decision
from .illustrated import asset_paths, prepare_brand


def verify(directory, top_k, style='classic', alignment_python=None):
    suffix='_lively' if style=='lively' else ''
    manifest=read_json(directory/'match_manifest.json');decision=read_json(directory/'edit_decision.json')
    validate_decision(decision,manifest,directory,top_k)
    render=read_json(directory/f'render_report{suffix}.json')
    if len(render['clips'])!=top_k or [c['rank'] for c in render['clips']]!=list(range(top_k,0,-1)):
        raise ValueError('成片回合数量或播放顺序错误')
    if style=='lively':
        from .presentation import validate_timeline
        validate_timeline(render,decision)
    reports=[]
    for item in render.get('segments',render['clips'])+[{'path':render['output'],'duration_sec':render['expected_duration_sec']}]:
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
        log=directory/f'verification{suffix}'/f'{path.stem}.log'
        run(['ffmpeg','-v','error','-xerror','-i',path,'-f','null','-'],log)
        reports.append({'path':str(path),'sha256':digest(path),'duration_sec':float(video['duration']),
                        'av_duration_delta_sec':av_delta,'av_start_delta_sec':av_start_delta,
                        'duration_error_sec':duration_error,'full_decode':'passed','frames':int(video['nb_frames'])})
    report={'status':'passed','top_k':top_k,'videos':reports,'source_has_audio':manifest['source']['has_audio'],
            'ranking_mode':decision['ranking_mode'],'note':'完整解码与时间轴校验不等于语义识别准确率。'}
    artifacts=[directory/f'verification{suffix}.json']
    if alignment_python:
        run([alignment_python,'-m','volleymole.check_alignment','--run',directory,'--style',style],directory/f'alignment{suffix}.log')
        artifacts.append(directory/f'alignment_verification{suffix}.json')
    if style=='lively':
        run([alignment_python or sys.executable,'-m','volleymole.check_title_cards','--run',directory],
            directory/'title_card_readability.log')
        artifacts.append(directory/'title_card_readability.json')
    save_json(directory/f'verification{suffix}.json',report)
    return str(directory/f'verification{suffix}.json'),artifacts


def main(argv=None):
    invocation_begin=time.perf_counter()
    started_at=datetime.now(timezone.utc).isoformat()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--top-k',type=int,choices=(5,10),default=5)
    parser.add_argument('--focus-player',type=int)
    parser.add_argument('--format',choices=['vertical'],default='vertical')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--analytics-cache',type=Path)
    parser.add_argument('--tracking-cache',type=Path)
    parser.add_argument('--evidence-cache',type=Path,help='显式导入同源已校验的 VolleyMole 运行目录')
    parser.add_argument('--models',type=Path)
    parser.add_argument('--inference-mode',choices=['shared','independent'],default='shared')
    parser.add_argument('--analysis-cache-dir',type=Path,default=ROOT/'runs/.analysis-cache')
    parser.add_argument('--no-analysis-cache',action='store_true',help='强制重新推理，不复用全场分析缓存')
    parser.add_argument('--pipeline-depth',type=int,choices=range(1,5),default=1)
    parser.add_argument('--auxiliary-device',help='辅助球检测设备，须属于 --devices')
    parser.add_argument('--vball-engine',choices=['ort','ort-bound'],default='ort')
    parser.add_argument('--preview-scope',choices=['needed','all'],default='needed')
    parser.add_argument('--preview-workers',type=int,choices=range(1,5),default=2)
    parser.add_argument('--render-workers',type=int,choices=range(1,5),default=1)
    parser.add_argument('--device',choices=['auto','cpu','cuda','cuda:0','cuda:1','cuda:2','cuda:3'],default='auto')
    from .gpu_stages import parse_devices
    parser.add_argument('--devices',type=parse_devices,
                        help='四卡共享推理：cuda:0,cuda:1,cuda:2,cuda:3（状态/动作/人物/球轨迹）')
    parser.add_argument('--ranker',choices=['auto','rules'],default='auto')
    parser.add_argument('--api-base',default=os.getenv('VOLLEYMOLE_API_BASE','https://api.openai.com/v1'))
    parser.add_argument('--model',default=os.getenv('VOLLEYMOLE_MODEL'))
    parser.add_argument('--vision-model',default=os.getenv('VOLLEYMOLE_VISION_MODEL'),help='可选独立视觉评审模型；主模型不支持图像时默认从同一服务自动选择')
    parser.add_argument('--api-timeout',type=float,default=240)
    parser.add_argument('--llm-config',type=Path,default=ROOT/'llm_api.json',help='读取 llm 节点；rank 节点为文本 reranker，不用于生成剪辑单')
    parser.add_argument('--font',default=str(DEFAULT_FONT))
    parser.add_argument('--style',choices=['classic','lively'],default='lively',help='lively：生图插画、手写标题、5秒快切、慢回放与3秒标题转场；classic：原版')
    parser.add_argument('--stop-after',choices=['manifest','rank','render'],default='render')
    parser.add_argument('--rerun-from',choices=['inference','analytics','tracking','player','rallies','previews','rank','render','verify'])
    args=parser.parse_args(argv)
    if args.auxiliary_device and (not args.devices or args.auxiliary_device not in args.devices):
        parser.error('--auxiliary-device 必须属于 --devices')
    performance={'pipeline_depth':args.pipeline_depth,'auxiliary_device':args.auxiliary_device,'vball_engine':args.vball_engine}
    if (args.inference_mode!='shared' or args.evidence_cache or args.analytics_cache or args.tracking_cache) and (
            args.pipeline_depth!=1 or args.auxiliary_device or args.vball_engine!='ort'):
        parser.error('推理优化选项仅适用于共享新鲜推理/共享缓存')
    if args.devices and (args.device!='auto' or args.inference_mode!='shared' or
                         args.evidence_cache or args.analytics_cache or args.tracking_cache):
        parser.error('--devices 仅适用于共享推理，不能与 --device、independent 或显式证据导入同时使用')
    args.analytics_python=args.tracking_python=args.player_python=sys.executable
    registry=ModelRegistry(args.models)
    os.environ['VOLLEYMOLE_MODELS']=str(registry.directory)
    suffix='_lively' if args.style=='lively' else ''
    render_stage='render'+suffix;verify_stage='verify'+suffix
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
    def finish_timing(output=None):
        total=time.perf_counter()-invocation_begin
        reused=[r['stage'] for r in stages.current_run if r['status']=='reused']
        data={'started_at_utc':started_at,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
              'style':args.style,'source_duration_sec':meta['duration_sec'],'total_elapsed_sec':round(total,3),
              'stages':stages.current_run,'reused_stages':reused,
              'preparation_and_bookkeeping_sec':round(max(0,total-sum(r['elapsed_sec'] for r in stages.current_run)),3),
              'output':str(output) if output else None,
              'scope_note':'本次命令墙钟耗时，包含输入检查、缓存校验、实际执行阶段及成片校验；不包含开发调试。复用的历史推理/API耗时不计入本次。',
              'upstream_modes':{name:read_json(directory/name/'provenance.json').get('mode','optional') for name in ('analytics','tracking','player') if (directory/name/'provenance.json').exists()}}
        if (directory/'inference_cache.json').is_file() and any(r['stage']=='inference' for r in stages.current_run):
            cache=read_json(directory/'inference_cache.json')
            fresh='inference' not in reused and cache['cache_status']!='reused'
            data['analysis_cache']={'fresh_inference_this_command':fresh,'fingerprint':cache['fingerprint'],
                'historical_inference_sec':cache['historical_inference_telemetry']['wall_sec'],
                'note':'Historical inference time is informational and excluded from this command when reused.'}
        if output:
            rendered=read_json(directory/f'render_report{suffix}.json')
            data['output_duration_sec']=rendered['expected_duration_sec']
            if render_stage not in reused:data['render_breakdown_sec']=rendered.get('render_timing_sec')
        tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        save_json(directory/'timings'/f'{tag}-{args.style}.json',data)
        save_json(directory/'timing_latest.json',data)
        if output and render_stage not in reused:save_json(directory/f'timing{suffix}.json',data)
        print(f'本次命令耗时：{total:.2f} 秒；复用阶段：{", ".join(reused) or "无"}',flush=True)
    if args.rerun_from:
        order=['inference','analytics','tracking','player','rallies','previews','rank',render_stage,verify_stage]
        target={'render':render_stage,'verify':verify_stage}.get(args.rerun_from,args.rerun_from)
        if target in ('analytics','tracking','player') and args.inference_mode=='shared':
            target='inference'
        for name in order[order.index(target):]:
            stages.data['stages'].pop(name,None)
        save_json(stages.path,stages.data)
    cached={k:args.evidence_cache.resolve()/k for k in ('analytics','tracking')} if args.evidence_cache else {}
    analytics_cache=args.analytics_cache or cached.get('analytics')
    tracking_cache=args.tracking_cache or cached.get('tracking')
    save_json(directory/'run_config.json',{'video':str(video),'top_k':args.top_k,'focus_player':args.focus_player,
              'analytics_cache':str(analytics_cache) if analytics_cache else None,'tracking_cache':str(tracking_cache) if tracking_cache else None,
              'device':args.device,'devices':args.devices,'config':config,'code':code,'inference_mode':args.inference_mode,
              'models':str(registry.directory),'analysis_cache_dir':str(args.analysis_cache_dir),
              'performance':performance,'preview_scope':args.preview_scope,
              'preview_workers':args.preview_workers,'render_workers':args.render_workers})
    def cache_sig(path, files):
        return [identity(Path(path)/f) for f in files] if path else None
    if args.inference_mode=='shared' and not analytics_cache and not tracking_cache:
        signature=inference_signature(meta['identity'],registry,args.device,args.focus_player,config['player_confidence'],args.devices,performance)
        force=args.no_analysis_cache or args.rerun_from in ('inference','analytics','tracking','player')
        if force:stages.data['stages'].pop('inference',None)
        result=stages.execute('inference',signature,lambda:ingest_shared(video,directory,registry,signature,args.analysis_cache_dir,force))
        analytics,tracking,player=(result[k] for k in ('analytics','tracking','player'))
    else:
        analytics=stages.execute('analytics',{'source':meta['identity'],'cache':cache_sig(analytics_cache,['summary.json','detections.jsonl']),
                            'worker':code,'models':registry.entries,'adapter':code['adapters.py'],'python':args.analytics_python,'device':args.device},
                            lambda:ingest_analytics(video,directory,analytics_cache,args.analytics_python,args.device))
        tracking=stages.execute('tracking',{'source':meta['identity'],'cache':cache_sig(tracking_cache,['ball.csv','source_pts.csv','provenance.json']),
                           'adapter':code,'models':registry.entries,'python':args.tracking_python,'device':args.device},
                           lambda:ingest_tracking(video,directory,tracking_cache,args.tracking_python,args.device))
        player=stages.execute('player',{'source':meta['identity'],'number':args.focus_player,'threshold':config['player_confidence'],
                         'worker':code,'models':registry.entries,'adapter':code['adapters.py'],'python':args.player_python,'device':args.device},
                         lambda:ingest_player(video,directory,args.focus_player,args.player_python,args.device,config['player_confidence']))
    def make_manifest():
        data,files=build_manifest(meta,analytics,tracking,directory/'tracking/source_pts.csv',player,directory,config,
                                 {k:read_json(directory/k/'provenance.json') for k in ('analytics','tracking','player')})
        return str(directory/'match_manifest.json'),files
    manifest_path=stages.execute('rallies',{'analytics':digest(analytics),'tracking':digest(tracking),'pts':digest(directory/'tracking/source_pts.csv'),
                                'player':digest(player),'config':config,'code':code['rally.py'],
                                'evidence_code':code['rally_evidence.py'],'source':meta},make_manifest)
    if args.stop_after=='manifest':finish_timing();return
    manifest=read_json(manifest_path)
    preview_ids=[r['rally_id'] for r in manifest['rallies']]
    if args.preview_scope=='needed':
        candidates=shortlist(manifest,args.top_k)
        if args.ranker=='rules':
            # Pure deterministic selection, before strict file validation in rank().
            preview_ids=[r['rally_id'] for r in rule_decision(candidates,args.top_k)['selected']]
        else:
            preview_ids=[r['rally_id'] for r in candidates]
    def make_previews():
        run([args.tracking_python,'-m','volleymole.media_worker','previews','--run',directory,
             '--workers',args.preview_workers,'--rally-ids',*preview_ids],directory/'previews.log')
        index=read_json(directory/'previews/index.json')
        return str(directory/'previews/index.json'),[directory/'previews/index.json']+[directory/p for p in index['files']]
    preview_signature={'source':meta['identity'],'samples':[(r['preview_times_sec'],r['preview_frames']) for r in manifest['rallies'] if r['rally_id'] in preview_ids],
                       'code':functions_digest(APP/'media_worker.py',{'previews','frame_at'})}
    stages.execute('previews',preview_signature,make_previews)
    decision=stages.execute('rank',{'manifest':digest(manifest_path),'k':args.top_k,'focus':args.focus_player,'ranker':args.ranker,
                            'model':args.model,'endpoint':args.api_base,'has_key':bool(os.getenv('VOLLEYMOLE_API_KEY') or os.getenv('OPENAI_API_KEY')),
                            'code':code['ranker.py'],'schema':code['schemas.py'],'semantic':code['semantic.py'],'vision_model':args.vision_model,
                            'prompt':digest(APP/'prompts/rank_top_plays.md'),'vision_prompt':digest(APP/'prompts/review_frames.md')},
                            lambda:rank(manifest,directory,args.top_k,args.focus_player,args.ranker,args.api_base,args.model,args.api_timeout,args.vision_model))
    if args.stop_after=='rank':finish_timing();return
    def make_video():
        run([args.tracking_python,'-m','volleymole.media_worker','render','--run',directory,'--font',args.font,'--style',args.style,
             '--workers',args.render_workers],directory/f'render{suffix}.log')
        report=read_json(directory/f'render_report{suffix}.json')
        return report['output'],[report['output'],directory/f'render_report{suffix}.json']+[c['path'] for c in report.get('segments',report['clips'])]
    if args.style=='lively':prepare_brand()
    output=stages.execute(render_stage,{'decision':digest(decision),'manifest':digest(manifest_path),'code':code['media_worker.py'],
                          'render_workers':args.render_workers,
                          'presentation':code['presentation.py'],'camera':code['camera.py'],'style':args.style,
                          'illustrated':code['illustrated.py'],
                          'assets':[identity(p) for p in asset_paths()] if args.style=='lively' else [],
                          'schema':code['schemas.py'],'font':identity(args.font)},make_video)
    stages.execute(verify_stage,{'output':digest(output),'decision':digest(decision),'report':digest(directory/f'render_report{suffix}.json'),
                             'code':code['run_match.py'],'alignment':code['check_alignment.py'],
                             'title_cards':digest(APP/'check_title_cards.py') if args.style=='lively' else None},
                   lambda:verify(directory,args.top_k,args.style,args.tracking_python))
    finish_timing(output)
    print(f'VolleyMole 成片：{output}',flush=True)


def verify_cli(argv=None):
    parser=argparse.ArgumentParser(description='Verify rendered output against source picture/audio')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--style',choices=['classic','lively'],default='lively')
    args=parser.parse_args(argv)
    directory=args.run.resolve()
    k=len(read_json(directory/'edit_decision.json')['selected'])
    verify(directory,k,args.style,sys.executable)
    print('All output checks passed')


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(f'VolleyMole: {exc}',file=sys.stderr);sys.exit(1)
