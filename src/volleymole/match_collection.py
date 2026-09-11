"""Date-grouped matches: analyze sets independently, rank all rallies together."""
import copy
from datetime import date
import fcntl
import os
from pathlib import Path
import re
import sys
import time
from .common import APP,ROOT,Stages,digest,read_json,save_json,run
from .run_match import argument_parser,verify,validate_event_arguments
from .sources import source_for

VIDEO_EXTENSIONS={'.mp4','.mov','.mkv','.avi','.m4v','.webm','.mts','.m2ts'}
NAME=re.compile(r'^(\d{4})\.(\d{1,2})\.(\d{1,2})\.(\d+)$')


def parse_name(path):
    match=NAME.fullmatch(Path(path).stem)
    if not match:raise ValueError(f'视频须命名为 年.月.日.局号，例如 2026.1.6.1.mp4：{Path(path).name}')
    year,month,day,number=map(int,match.groups())
    if number<1:raise ValueError('局号必须从 1 开始')
    return date(year,month,day).isoformat(),number


def discover_matches(directory):
    directory=Path(directory).resolve()
    if not directory.is_dir():raise ValueError(f'素材目录不存在：{directory}')
    groups={}
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:continue
        day,number=parse_name(path)
        group=groups.setdefault(day,{})
        if number in group:raise ValueError(f'{day} 的第 {number} 局重复：{group[number].name} / {path.name}')
        group[number]=path
    if not groups:raise ValueError('目录中没有支持的视频')
    return {day:sorted(group.items()) for day,group in sorted(groups.items())}


def merge_manifests(parts,directory,match_date):
    """No video concat or clock shifting: IDs and file paths are namespaced."""
    sources={};rallies=[];config=None
    for number,part in parts:
        manifest=read_json(part/'match_manifest.json')
        if 'sources' in manifest:raise ValueError('分局输入不能再次包含整场合并清单')
        if config is not None and config!=manifest['config']:raise ValueError('各局分析配置不一致，不能比较回合分数')
        config=manifest['config']
        prefix=part.relative_to(directory).as_posix()
        key=f'set-{number:04d}'
        sources[key]={**manifest['source'],'set_number':number,'evidence_root':prefix,
                      'manifest_sha256':digest(part/'match_manifest.json')}
        for original in manifest['rallies']:
            r=copy.deepcopy(original)
            r.update(rally_id=f'{key}__{original["rally_id"]}',original_rally_id=original['rally_id'],
                     source_id=key,source_set=number)
            for name in [r['tracking_json'],*r['preview_frames']]:
                if not (part/name).resolve().is_relative_to(part.resolve()):raise ValueError('分局证据路径越界')
            r['tracking_json']=f'{prefix}/{r["tracking_json"]}'
            r['preview_frames']=[f'{prefix}/{p}' for p in r['preview_frames']]
            for field in ('analytics_jsonl','ball_csv'):
                if field in r.get('evidence',{}):r['evidence'][field]=f'{prefix}/{r["evidence"][field]}'
            rallies.append(r)
    if not sources:raise ValueError('整场必须包含至少一局')
    return {'match_date':match_date,'time_basis':'source-local seconds; resolve rally.source_id before seeking',
            'source':{'kind':'multi_video','duration_sec':sum(s['duration_sec'] for s in sources.values()),
                      'has_audio':any(s['has_audio'] for s in sources.values()),'set_count':len(sources)},
            'sources':sources,'config':config,'rallies':rallies,
            'ranking_scope':'all eligible rallies across all sets; no per-set highlight quota'}


def part_options(args):
    options=[]
    names=('models','config','focus_player','inference_mode','analysis_cache_dir','device','llm_config',
           'pipeline_depth','auxiliary_device','vball_engine','preview_workers')
    for name in names:
        value=getattr(args,name)
        if value is not None:options.extend(['--'+name.replace('_','-'),str(value)])
    if args.devices:options.extend(['--devices',','.join(args.devices)])
    if args.no_analysis_cache:options.append('--no-analysis-cache')
    return options


def execute_match(args,day,sets):
    from .ranker import rank,shortlist,rule_decision
    from .media_worker import previews,render
    from .design_suites import resolve_design
    from .illustrated import asset_paths
    directory=(args.output or ROOT/'runs/matches')/f'{day}-top{args.top_k}'
    directory=directory.resolve();directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    llm=read_json(args.llm_config).get('llm',{}) if args.ranker!='rules' and args.llm_config.is_file() else {}
    if llm.get('provider') not in (None,'openai_compatible'):raise ValueError('只支持 openai_compatible 接口')
    if llm.get('api_key'):os.environ['VOLLEYMOLE_API_KEY']=llm['api_key']
    args.api_base=llm.get('base_url',args.api_base)
    args.model=args.model or llm.get('model');args.vision_model=args.vision_model or llm.get('vision_model')
    timelines=[]
    with (directory/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'整场目录正在使用：{directory}')
        parts=[]
        for number,path in sets:
            part=directory/'parts'/f'set-{number:04d}';part.mkdir(parents=True,exist_ok=True)
            print(f'[{day}] 分析第 {number} 局：{path.name}',flush=True)
            discovery=None
            event_options=[]
            if args.ranker!='rules' and args.stop_after!='manifest':
                from .common import probe,identity
                from .event_pipeline import Discovery
                source=probe(path);source['identity']=identity(path)
                frame_cache=part/'event_frames' if args.inference_mode=='shared' else None
                if frame_cache:
                    save_json(frame_cache/'status.json',{'status':'pending'})
                    event_options=['--event-frame-cache',frame_cache,'--review-fps',args.review_fps]
                discovery=Discovery(source,part,args,deadline=started+args.analysis_timeout,frame_cache=frame_cache)
            try:
                run([sys.executable,'-m','volleymole.run_match','--video',path,'--output',part,
                     '--stop-after','manifest','--ranker','rules',*part_options(args),*event_options],part/'match_analysis.log')
            except BaseException:
                if discovery:
                    if frame_cache: save_json(frame_cache/'status.json',{'status':'unavailable'})
                    discovery.close()
                raise
            parts.append((number,part))
            if discovery:
                timelines.append((number,discovery.finish(read_json(part/'match_manifest.json'))))
        manifest=merge_manifests(parts,directory,day)
        save_json(directory/'match_manifest.json',manifest)
        stages=Stages(directory)
        if args.rerun_from:
            targets=['previews','rank','render','verify']
            target=args.rerun_from
            if target not in targets:raise ValueError('整场 --rerun-from 支持 previews/rank/render/verify；重推理用 --no-analysis-cache')
            for name in targets[targets.index(target):]:stages.data['stages'].pop(name,None)
        art,title,transition=resolve_design(args.design_suite,args.design_language,args.art_theme,args.title_template,args.transition_style)
        save_json(directory/'run_config.json',{'mode':'date_grouped_match','date':day,'top_k':args.top_k,
            'quality':args.quality,'render_workers':args.render_workers,'design_suite':args.design_suite,'design_language':args.design_language,
            'art_theme':art,'title_template':title,'transition_style':transition,'style':args.style})
        if args.stop_after=='manifest':return {'date':day,'directory':str(directory),'status':'manifest_only'}
        if args.ranker!='rules':
            from .event_pipeline import complete_collections
            from .events import merge_events
            timeline={'schema_version':1,'events':merge_events([{**e,'source_id':f'set-{n:04d}'}
                for n,t in timelines for e in t['events']]),'sets':[{'set_number':n,**{k:v for k,v in t.items() if k not in ('events','source')}} for n,t in timelines]}
            save_json(directory/'event_timeline.json',timeline)
            args.art_theme,args.title_template,args.transition_style=art,title,transition
            collections=complete_collections(args,manifest,timeline,directory)
            result={'date':day,'directory':str(directory),'status':'rank_only' if args.stop_after=='rank' else 'collections_completed',
                'collections':collections,'elapsed_sec':round(time.monotonic()-started,3)}
            save_json(directory/'match_summary.json',result)
            return result
        candidates=shortlist(manifest,args.top_k)
        chosen=rule_decision(candidates,args.top_k)['selected'] if args.ranker=='rules' else candidates
        ids=[r['rally_id'] for r in chosen]
        if args.preview_scope=='all':ids=[r['rally_id'] for r in manifest['rallies']]
        code={p.name:digest(p) for p in APP.glob('*.py')}
        signature={'manifest':digest(directory/'match_manifest.json'),'code':code,'top_k':args.top_k,'mode':args.ranker}
        def make_previews():
            previews(directory,ids,args.preview_workers)
            index=directory/'previews/index.json'
            return str(index),[index]+[directory/p for p in read_json(index)['files']]
        stages.execute('previews',{**signature,'ids':ids},make_previews)
        llm=read_json(args.llm_config).get('llm',{}) if args.ranker!='rules' and args.llm_config.is_file() else {}
        if llm.get('provider') not in (None,'openai_compatible'):raise ValueError('只支持 openai_compatible 接口')
        if llm.get('api_key'):os.environ['VOLLEYMOLE_API_KEY']=llm['api_key']
        endpoint=llm.get('base_url',args.api_base);model=args.model or llm.get('model');vision=args.vision_model or llm.get('vision_model')
        stages.execute('rank',{**signature,'focus':args.focus_player,'endpoint':endpoint,'model':model,'vision':vision,
            'has_key':bool(os.getenv('VOLLEYMOLE_API_KEY') or os.getenv('OPENAI_API_KEY'))},
            lambda:rank(manifest,directory,args.top_k,args.focus_player,args.ranker,endpoint,model,args.api_timeout,vision))
        decision=read_json(directory/'edit_decision.json')
        selected=[]
        for item in decision['selected']:
            source=source_for(manifest,item['rally_id'])
            selected.append({**item,'source_path':source['path'],'set_number':source['set_number']})
        save_json(directory/'selected_sources.json',{'date':day,'time_basis':'source-local seconds','selected':selected})
        if args.stop_after=='rank':return {'date':day,'directory':str(directory),'status':'rank_only'}
        suffix='_lively' if args.style=='lively' else ''
        report_path=directory/f'render_report{suffix}.json'
        render_sig={**signature,'decision':digest(directory/'edit_decision.json'),'config':read_json(directory/'run_config.json'),
                    'font':digest(args.font),'assets':[digest(p) for p in asset_paths(art,title,transition,args.design_suite)]}
        def make_video():
            render(directory,args.font,args.style,args.render_workers,art,title,transition,args.design_suite,args.design_language,args.quality)
            report=read_json(report_path)
            return report['output'],[report_path,report['output']]+[c['path'] for c in report.get('segments',report['clips'])]
        output=stages.execute('render',render_sig,make_video)
        stages.execute('verify',{'output':digest(output),'report':digest(report_path),'code':code},
                       lambda:verify(directory,args.top_k,args.style,sys.executable))
        result={'date':day,'status':'passed','sets':[{'number':n,'path':str(p)} for n,p in sets],
                'eligible_rallies':sum(r['eligible'] for r in manifest['rallies']),'top_k':args.top_k,'output':output,
                'elapsed_sec':round(time.monotonic()-started,3),'stages':stages.current_run}
        save_json(directory/'match_summary.json',result)
        return result


def main(argv=None):
    parser=argument_parser();parser.prog='volleymole match'
    parser.description='按 年.月.日.局号 分组，各局独立分析，整场统一排名；默认十佳球'
    for action in parser._actions:
        if action.dest=='video':action.required=False
    parser.add_argument('--input-dir',type=Path,required=True,help='按日期分组的视频目录')
    parser.add_argument('--date',help='仅处理指定日期，例如 2026.1.6 或 2026-01-06；省略则处理全部日期')
    parser.add_argument('--list',action='store_true',help='仅列出日期与局号，不运行推理或写文件')
    parser.set_defaults(top_k=10)
    args=parser.parse_args(argv)
    validate_event_arguments(args,parser)
    if args.video or args.evidence_cache or args.analytics_cache or args.tracking_cache:
        parser.error('整场模式使用 --input-dir 与共享分析缓存，不接受单原片或单局显式证据导入')
    if args.rerun_from and args.rerun_from not in ('previews','rank','render','verify'):
        parser.error('整场 --rerun-from 支持 previews/rank/render/verify；重推理用 --no-analysis-cache')
    if args.style!='lively' and (args.design_suite!='custom' or args.art_theme!='default' or args.title_template!='legacy' or args.transition_style!='fade'):
        parser.error('设计套装和标题/转场选项只适用于 lively')
    groups=discover_matches(args.input_dir)
    if args.date:
        try:day=date(*map(int,re.split(r'[.-]',args.date))).isoformat()
        except (ValueError,TypeError):parser.error('日期格式应为 2026.1.6 或 2026-01-06')
        if day not in groups:parser.error(f'未找到该日期：{day}')
        groups={day:groups[day]}
    for day,sets in groups.items():
        print(f'{day}: '+', '.join(f'第 {n} 局 {p.name}' for n,p in sets),flush=True)
        missing=sorted(set(range(1,max(n for n,_ in sets)+1))-{n for n,_ in sets})
        if missing:print(f'提示：缺少局号 {missing}，只处理目录中实际存在的局。',flush=True)
    if args.list:return
    results=[]
    for day,sets in groups.items():
        results.append(execute_match(args,day,sets))
        save_json((args.output or ROOT/'runs/matches')/'matches_report.json',{'matches':results})
        print(f'[{day}] {results[-1]["status"]}: {results[-1].get("output",results[-1].get("directory"))}',flush=True)


if __name__=='__main__':main()
