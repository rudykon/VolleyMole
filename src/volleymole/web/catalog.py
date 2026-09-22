"""Expose the actual CLI parsers as forms; never accept shell commands."""
import argparse
import contextlib
import importlib
import io
import math
from pathlib import Path

COMMANDS = {
    'run': ('单视频剪辑', 'run_match', 'argument_parser'),
    'match': ('整场多局合辑', 'match_collection', 'match_argument_parser'),
    'infer': ('独立模型推理', 'inference', 'argument_parser'),
    'verify': ('成片完整性验证', 'run_match', 'verify_argument_parser'),
    'review-replays': ('慢回放复核', 'replay_stage', 'argument_parser'),
    'meme-audio': ('本地配音混音', 'meme_audio', 'argument_parser'),
    'meme-director': ('智能配音导演', 'meme_stage', 'argument_parser'),
    'models': ('模型管理', 'models', 'argument_parser'),
    'assets': ('视觉素材管理', 'assets', 'argument_parser'),
    'templates': ('模板导入与导出', 'templates', 'argument_parser'),
}
LABELS = {
    'video':'源视频', 'output':'输出位置', 'input_dir':'多局录像目录', 'date':'比赛日期',
    'top_k':'入选数量', 'collection':'榜单', 'analysis_mode':'分析方式', 'ranker':'排名方式',
    'template':'成片模板', 'design_suite':'视觉套装', 'design_language':'文字语言', 'quality':'输出画质',
    'focus_player':'关注球员号码', 'device':'计算设备', 'devices':'多卡设备', 'models':'模型目录',
    'style':'呈现模式', 'art_theme':'插画主题', 'title_template':'标题样式', 'transition_style':'转场样式',
    'replays':'启用慢回放', 'replay_speed':'慢回放速度', 'replay_review':'慢回放复核策略',
    'replay_review_timeout':'复核总时限（秒）', 'replay_review_concurrency':'回放复核并发数',
    'replay_scan_fps':'回放粗采样帧率', 'replay_review_fps':'回放精采样帧率',
    'replay_review_width':'回放采样宽度', 'replay_max_expansions':'回放边界扩展次数',
    'replace_replay_reviews':'重新执行已有复核', 'run':'运行目录', 'manifest':'成片清单',
    'plan':'配音计划', 'max_cues':'最多配音次数', 'rms_db':'配音音量（dB）',
    'duck_db':'原声降低（dB）', 'audio_enabled':'启用配音', 'max_calls':'API 调用上限',
    'render_output':'配音成片输出', 'api_base':'API 地址', 'model':'模型名称', 'vision_model':'视觉复核模型',
    'llm_config':'API 配置文件', 'api_timeout':'请求超时（秒）', 'protocol':'接口协议',
    'max_output_tokens':'最大输出 token', 'thinking_level':'推理强度',
    'semantic_concurrency':'语义请求并发数', 'semantic_max_tokens':'语义输出 token 上限',
    'semantic_frame_width':'语义采样宽度', 'semantic_reasoning_effort':'语义推理强度',
    'analysis_timeout':'单局分析时限（秒）', 'review_budget_fraction':'精细复核预算比例',
    'semantic_retries':'语义重试次数', 'event_chunk_sec':'事件分块时长（秒）',
    'event_overlap_sec':'分块重叠（秒）', 'coarse_fps':'粗读采样帧率', 'review_fps':'精读采样帧率',
    'max_review_sec':'最长复核片段（秒）', 'semantic_modality':'语义输入',
    'sound_model':'声音模型', 'sound_labels':'声音类别文件', 'sound_thresholds':'声音阈值文件',
    'no_sound_model':'关闭声音模型', 'sound_device':'声音模型设备',
    'action_evidence':'动作候选文件', 'action_evidence_dir':'动作候选缓存目录',
    'format':'画面比例', 'config':'算法配置文件', 'analytics_cache':'动作分析缓存',
    'tracking_cache':'跟踪缓存', 'evidence_cache':'已有证据目录', 'inference_mode':'推理方式',
    'analysis_cache_dir':'共享分析缓存目录', 'no_analysis_cache':'重新计算分析缓存',
    'pipeline_depth':'流水线深度', 'auxiliary_device':'辅助检测设备', 'vball_engine':'球跟踪引擎',
    'preview_scope':'预览范围', 'preview_workers':'预览并行数', 'render_workers':'渲染并行数',
    'font':'字体文件', 'stop_after':'完成至阶段', 'rerun_from':'从阶段重新执行', 'list':'仅列出比赛',
    'kind':'推理类型', 'number':'球衣号码', 'confidence':'识别置信度', 'max_frames':'最大帧数（测试用）',
    'half':'半精度推理', 'directory':'安装 / 模板目录', 'names':'模型名称（多个用空格分隔）',
    'name':'名称', 'source':'本地模型文件', 'archive':'素材 ZIP', 'file':'模板文件', 'from_run':'从运行配置导出',
}
GROUPS = {
    '基础设置': {'video','input_dir','date','top_k','collection','analysis_mode','ranker','output','template','focus_player','run','kind','manifest','plan','number','list'},
    '视觉与慢回放': {'style','art_theme','title_template','transition_style','design_suite','design_language','quality','format','font'},
    '计算与缓存': {'device','devices','models','inference_mode','analysis_cache_dir','no_analysis_cache','pipeline_depth','auxiliary_device','vball_engine','preview_scope','preview_workers','render_workers','analytics_cache','tracking_cache','evidence_cache','half','confidence','max_frames'},
    '事件与声音': {'analysis_timeout','review_budget_fraction','event_chunk_sec','event_overlap_sec','coarse_fps','review_fps','max_review_sec','semantic_modality','sound_model','sound_labels','sound_thresholds','no_sound_model','sound_device','action_evidence','action_evidence_dir'},
    'API 与请求': {'api_base','model','vision_model','llm_config','api_timeout','semantic_concurrency','semantic_max_tokens','semantic_frame_width','semantic_reasoning_effort','semantic_retries','protocol','max_output_tokens','thinking_level','max_calls'},
}
DIR_FIELDS = {'input_dir','run','models','directory','analysis_cache_dir','analytics_cache','tracking_cache','evidence_cache','action_evidence_dir','from_run'}


def parser_for(command):
    if command not in COMMANDS:
        raise ValueError('未知功能')
    _, module, factory = COMMANDS[command]
    return getattr(importlib.import_module('volleymole.' + module), factory)()


def branches(parser):
    sub = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
    return sub.choices if sub else {}


def fields(parser):
    rows = []
    for a in parser._actions:
        if a.dest == 'help' or isinstance(a, argparse._SubParsersAction) or a.help == argparse.SUPPRESS:
            continue
        kind = ('boolean' if isinstance(a, (argparse.BooleanOptionalAction, argparse._StoreTrueAction, argparse._StoreFalseAction))
                else 'integer' if a.type is int else 'number' if a.type is float else 'path' if a.type is Path or a.dest == 'font' else 'text')
        group = next((k for k, v in GROUPS.items() if a.dest in v), '其他设置')
        if a.dest.startswith('replay') or a.dest == 'replace_replay_reviews':
            group = '视觉与慢回放'
        rows.append({'key':a.dest, 'label':LABELS.get(a.dest,a.dest), 'type':kind,
                     'required':a.required, 'choices':list(a.choices) if a.choices is not None else None,
                     'default':str(a.default) if isinstance(a.default,Path) else a.default,
                     'help':a.help or '', 'group':group, 'multiple':a.nargs in ('*','+'),
                     'directory':a.dest in DIR_FIELDS, 'positional':not a.option_strings})
    return rows


def catalog():
    result=[]
    for command,(title,_,_) in COMMANDS.items():
        parser=parser_for(command)
        result.append({'id':command,'title':title,'fields':fields(parser),
                       'subcommands':[{'id':name,'fields':fields(p)} for name,p in branches(parser).items()]})
    return result


def build_argv(command, subcommand, values, path_check):
    """Strict typed values -> argv. argparse remains the source of truth."""
    if not isinstance(values,dict):
        raise ValueError('参数必须为对象')
    parser=parser_for(command)
    subs=branches(parser)
    if subs and subcommand not in subs:
        raise ValueError('请选择子功能')
    if not subs and subcommand:
        raise ValueError('此功能没有子命令')
    parsers=[parser]+([subs[subcommand]] if subs else [])
    allowed={f['key']:f for p in parsers for f in fields(p)}
    if set(values)-set(allowed):
        raise ValueError('未知或内部参数：'+', '.join(sorted(set(values)-set(allowed))))
    clean={}
    for key,value in values.items():
        if value is None or value == '':
            continue
        f=allowed[key]
        if f['type']=='boolean':
            if type(value) is not bool: raise ValueError(f'{key} 需要开关值')
        elif f['type'] in ('integer','number'):
            if type(value) not in (int,float) or not math.isfinite(value) or (f['type']=='integer' and type(value) is not int):
                raise ValueError(f'{key} 需要有效数值')
        elif f['multiple']:
            if not isinstance(value,list) or not all(isinstance(x,str) and x and not x.startswith('-') for x in value):
                raise ValueError(f'{key} 需要名称列表')
        elif not isinstance(value,str) or '\x00' in value or '\n' in value:
            raise ValueError(f'{key} 需要单行文本')
        if f['choices'] is not None and value not in f['choices']:
            raise ValueError(f'{key} 不在可选范围内')
        if f['type']=='path' or (key=='template' and ('/' in value or '\\' in value or value.endswith('.json'))):
            value=str(path_check(value,key))
        if f['positional'] and isinstance(value,str) and value.startswith('-'):
            raise ValueError('名称不能以 - 开头')
        clean[key]=value
    argv=[]
    for i,p in enumerate(parsers):
        if i: argv.append(subcommand)
        for a in p._actions:
            if a.dest not in clean: continue
            value=clean[a.dest]
            if isinstance(a,argparse.BooleanOptionalAction):
                argv.append(a.option_strings[0 if value else 1])
            elif isinstance(a,(argparse._StoreTrueAction,argparse._StoreFalseAction)):
                if value == a.const: argv.append(a.option_strings[0])
            elif a.option_strings:
                argv.append(a.option_strings[0]+'='+str(value))
            else:
                argv.extend(value if isinstance(value,list) else [str(value)])
    errors=io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            parser.parse_args(argv)
    except SystemExit:
        raise ValueError(errors.getvalue().strip().split('error:')[-1].strip()) from None
    return [command]+argv,clean
