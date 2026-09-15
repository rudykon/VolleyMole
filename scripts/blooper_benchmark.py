#!/usr/bin/env python3
"""Prepare blank blind review packages and evaluate imported human judgments."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from volleymole.blooper_benchmark import prepare, evaluate, _sha_file, VERSION, DIMENSIONS, FLAGS, STAGES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def assignment_page(form):
    """Embed only one sanitized, blank assignment; never read the private map."""
    condition = form['condition']
    if (form.get('schema_version') != VERSION or condition not in ('audio', 'silent')
            or not re.fullmatch(condition+r'_\d{2,}', form.get('assignment_id', ''))
            or not re.fullmatch(r'[a-f0-9]{64}', form.get('benchmark_id', ''))):
        raise ValueError('HTML 需要有效的独立盲评表')
    if form.get('human_provenance') != {'annotator_id': None, 'human_only': False}:
        raise ValueError('只从尚未填写的空表生成页面，不能预填身份或人工声明')
    rows, seen = [], set()
    for row in form['items']:
        iid = row.get('item_id', '')
        if (not re.fullmatch(r'[a-f0-9]{20}', iid) or iid in seen
                or row.get('media') != f'media/{iid}_{condition}.mp4'
                or type(row.get('duration_sec')) not in (int, float)
                or not math.isfinite(row['duration_sec']) or row['duration_sec'] <= 0):
            raise ValueError('HTML 只允许匿名本地素材及有效时长')
        seen.add(iid)
        if (row.get('completed') is not False or row.get('observed_facts') is not None
                or row.get('unknown_reason') is not None
                or row.get('ratings') != dict.fromkeys(DIMENSIONS)
                or row.get('flags') != dict.fromkeys(FLAGS)
                or row.get('stages') != {s: {'start_sec': None, 'end_sec': None} for s in STAGES}):
            raise ValueError('HTML 初始评分、边界和说明必须全空')
        # Allowlist protects against accidental private/model fields in an input.
        rows.append({k: row[k] for k in ('item_id', 'media', 'duration_sec', 'completed',
                                        'ratings', 'flags', 'stages', 'observed_facts', 'unknown_reason')})
    clean = {k: form[k] for k in ('schema_version', 'benchmark_id', 'assignment_id',
                                  'condition', 'human_provenance')}
    clean['items'] = rows
    if not rows:
        raise ValueError('HTML 评审表不能为空')
    embedded = json.dumps(clean, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return REVIEW_HTML.replace('__ASSIGNMENT_JSON__', embedded)


REVIEW_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file: blob:; connect-src 'none'">
<title>五大囧独立盲评</title>
<style>
body{margin:0;background:#f4f6f8;color:#17212b;font:16px/1.6 system-ui,sans-serif}
main{max-width:1060px;margin:auto;padding:24px}h1{font-size:25px}h2{font-size:19px;margin-top:24px}
section,header{background:white;border:1px solid #d8dee5;border-radius:8px;padding:18px;margin:16px 0}
video{display:block;width:100%;max-height:58vh;background:#111}label{display:block;margin:10px 0}
input,select,textarea,button{font:inherit}select,input[type=text],textarea{box-sizing:border-box;width:100%;padding:8px;border:1px solid #aab4bf;border-radius:4px}
input[type=number]{width:105px;padding:6px}textarea{min-height:86px}button{cursor:pointer;padding:8px 14px;border:1px solid #94a3b2;background:#fff;border-radius:5px}
button:disabled{opacity:.45;cursor:default}.primary{background:#175f9d;color:white}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 20px}
.muted{color:#566474;font-size:14px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}#error{color:#a31a24;white-space:pre-line}#progress{font-weight:600}
.stage{margin:14px 0}.stage label{display:inline-block;margin:4px 10px 4px 0}.stage button{font-size:13px;padding:5px}details{margin:12px 0}
@media(max-width:650px){main{padding:12px}.grid{grid-template-columns:1fr}section,header{padding:13px}}
</style></head><body><main>
<header><h1>五大囧独立盲评</h1>
<p id="assignment"></p><p>先完整观看原速视频，记录直接观察，再独立给总体分。分析维度不用于自动算总体分。未知与 0 分不同。</p>
<p class="muted">内容仅保存在当前页面内存。离开前下载 JSON；本页不联网、不自动评分。请独立填写，不查看其他评员答案或模型结果。</p>
<label>本人匿名评员 ID（同一个人始终用同一 ID）<input id="person" type="text" maxlength="200" autocomplete="off"></label>
<label><input id="human" type="checkbox"> 我确认由本人独立观看并填写，未以模型结果代替人工判断。</label>
</header>
<section><div class="toolbar"><button id="previous" type="button">上一段</button><span id="progress"></span><button id="next" type="button">下一段</button></div>
<p id="clip-name" class="muted"></p><video id="player" controls preload="metadata" playsinline></video><p id="clock" class="muted">片段时间 0.00 秒</p>
<label>直接观察（动作前因、结果及人物反应；不要根据常见套路补全）<textarea id="facts"></textarea></label>
<h2>铺垫 → 意外 → 反应</h2><p class="muted">时间从本片段 0 秒开始。每阶段起止一起填写；无法确定就都留空。反应可与意外交叠。</p><div id="stages"></div>
<h2>独立总体判断</h2><div id="overall" class="grid"></div>
<h2>分析维度</h2><p id="sound-note" class="muted"></p><div id="dimensions" class="grid"></div>
<h2>事实与展示适合性</h2><div id="flags" class="grid"></div>
<label>未知原因、缺失证据或争议（包括此前见过该片段）<textarea id="unknown"></textarea></label>
<label><input id="completed" type="checkbox"> 我已完整查看并完成本段判断；仍未知的字段保留为空。</label>
</section>
<section><div class="toolbar"><button id="download" class="primary" type="button">下载评分 JSON / 草稿</button><span id="status" class="muted"></span></div><p id="error" role="alert"></p>
<p class="muted">未完成的项目可以保存为草稿。下载不等于提交；把下载文件交回项目策划者。不要更改评审项目顺序。</p></section>
</main><script id="assignment-data" type="application/json">__ASSIGNMENT_JSON__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('assignment-data').textContent);
let index=0,dirty=false;
const $=id=>document.getElementById(id);
const ratingSpecs={
 overall_fun:['总体有趣度',['完整过程没有可解释趣味','有轻微趣味','趣味明确，力度一般','明显有趣，值得分享','趣味鲜明，强烈且可复述']],
 selection_suitability:['五大囧入选适合度',['不宜入选','不推荐','可作备选','推荐入选','强烈推荐优先入选']],
 unexpected_contrast:['意外与反差',['正常预期，无反差','小偏差','明确动作/结果反差','明显出人意料且过程可辨','多步反转或罕见反差，证据完整']],
 related_laughter:['相关笑声',['音轨清楚，完整观察到无相关笑声','可确认相关的轻微短笑','少数人清晰发笑','多人明显发笑且同步','持续集体发笑，归属清楚']],
 narrative:['事件叙事完整性',['无法建立前后关系','只见突发瞬间','可解释意外但缺必要前后文','铺垫、意外、后果完整','完整紧凑，无需猜测']],
 player_reaction:['球员与同伴反应',['人物可见、观察充分而无明显反应','轻微个人表情/动作','清晰单人或两人反应','多人明显互动','多人持续来回互动，强化趣味']]
};
const flagNames={event_visible:'画面足以看清本事件',boundary_complete:'解释事件所需开头结尾完整',injury_suspected:'有疑似受伤或疼痛迹象',safe_to_include:'适合作为趣味集锦公开展示',laughter_linked:'笑声可归属于此事件',ordinary_error:'仅为普通失误，无明确特殊反差',unrelated_laughter:'有笑声但与本事件无关'};
const stageNames={setup:'铺垫',unexpected:'意外',reaction:'反应'};
const soundKeys=new Set(['related_laughter','laughter_linked','unrelated_laughter']);
const refs={ratings:{},flags:{},stages:{}};
function changed(){dirty=true;$('status').textContent='有未下载的编辑';$('error').textContent='';}
function selectField(parent,key,title,options,group){
 const label=document.createElement('label');label.textContent=title;
 const select=document.createElement('select');select.id=group+'-'+key;
 for(const [value,text] of [['','未填写 / 未知'],...options]){const opt=document.createElement('option');opt.value=value;opt.textContent=text;select.append(opt);}
 select.disabled=data.condition==='silent'&&soundKeys.has(key);
 select.addEventListener('change',()=>{data.items[index][group][key]=select.value===''?null:(group==='ratings'?Number(select.value):select.value==='true');changed();});
 refs[group][key]=select;label.append(select);$(parent).append(label);
}
for(const [key,[title,anchors]] of Object.entries(ratingSpecs))selectField(['overall_fun','selection_suitability'].includes(key)?'overall':'dimensions',key,title,anchors.map((text,i)=>[String(i),i+' · '+text]),'ratings');
for(const [key,title] of Object.entries(flagNames))selectField('flags',key,title,[['true','是'],['false','否']],'flags');
for(const [stage,title] of Object.entries(stageNames)){
 const group=document.createElement('div');group.className='stage';const caption=document.createElement('strong');caption.textContent=title+' ';group.append(caption);refs.stages[stage]={};
 for(const [field,text] of [['start_sec','起点'],['end_sec','终点']]){
  const label=document.createElement('label');label.textContent=text+' ';const input=document.createElement('input');input.type='number';input.step='0.01';input.min='0';input.id=stage+'-'+field;
  input.addEventListener('input',()=>{data.items[index].stages[stage][field]=input.value===''?null:input.valueAsNumber;changed();});
  const button=document.createElement('button');button.type='button';button.textContent='填当前时间';
  button.addEventListener('click',()=>{const t=Math.min(data.items[index].duration_sec,Math.max(0,$('player').currentTime));input.value=t.toFixed(2);input.dispatchEvent(new Event('input'));});
  refs.stages[stage][field]=input;label.append(input);group.append(label,button);
 }
 $('stages').append(group);
}
$('assignment').textContent='独立评审表 '+data.assignment_id+' · '+(data.condition==='silent'?'静音条件':'原声音条件');
$('sound-note').textContent=data.condition==='silent'?'静音条件的笑声及音画归属无法判断，相关字段已禁用并保留未知。':'只有音轨足够清楚且观察完整，才能把“无相关笑声”填为 0；归属不清填未知。';
function refresh(){
 const row=data.items[index];$('progress').textContent=(index+1)+' / '+data.items.length;$('previous').disabled=index===0;$('next').disabled=index===data.items.length-1;
 $('clip-name').textContent='匿名片段 '+row.item_id+' · '+row.duration_sec.toFixed(2)+' 秒';
 $('player').pause();$('player').src=row.media;$('player').muted=data.condition==='silent';$('player').playbackRate=1;$('clock').textContent='片段时间 0.00 秒';
 for(const group of ['ratings','flags'])for(const [key,input] of Object.entries(refs[group]))input.value=row[group][key]===null?'':String(row[group][key]);
 for(const stage of Object.keys(stageNames))for(const field of ['start_sec','end_sec']){const input=refs.stages[stage][field];input.value=row.stages[stage][field]===null?'':row.stages[stage][field];input.max=row.duration_sec;}
 $('facts').value=row.observed_facts??'';$('unknown').value=row.unknown_reason??'';$('completed').checked=row.completed;
}
$('previous').addEventListener('click',()=>{index--;refresh();});$('next').addEventListener('click',()=>{index++;refresh();});
$('player').addEventListener('timeupdate',()=>{$('clock').textContent='片段时间 '+$('player').currentTime.toFixed(2)+' 秒';});
$('person').addEventListener('input',()=>{data.human_provenance.annotator_id=$('person').value.trim()||null;changed();});
$('human').addEventListener('change',()=>{data.human_provenance.human_only=$('human').checked;changed();});
for(const [id,key] of [['facts','observed_facts'],['unknown','unknown_reason']])$(id).addEventListener('input',()=>{data.items[index][key]=$(id).value.trim()||null;changed();});
$('completed').addEventListener('change',()=>{data.items[index].completed=$('completed').checked;changed();});
function validationError(){
 for(let i=0;i<data.items.length;i++){
  const row=data.items[i],prefix='第 '+(i+1)+' 段：';let previous=-1;
  for(const stage of Object.keys(stageNames)){
   const a=row.stages[stage].start_sec,b=row.stages[stage].end_sec;
   if((a===null)!==(b===null))return prefix+stageNames[stage]+'起止须一起填写或一起留空。';
   if(a!==null){if(!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<a||b>row.duration_sec)return prefix+'时间须位于片段内，终点不得早于起点。';if(a<previous)return prefix+'铺垫、意外、反应的起点顺序不成立。';previous=a;}
  }
  if(!row.completed)continue;
  if(!data.human_provenance.annotator_id||!data.human_provenance.human_only)return '已完成项需要填写本人匿名 ID，并主动确认独立真人声明。';
  if(!row.observed_facts)return prefix+'请写直接观察。';
  const unknown=Object.values(row.ratings).includes(null)||Object.values(row.flags).includes(null)||Object.values(row.stages).some(s=>s.start_sec===null);
  if(unknown&&!row.unknown_reason)return prefix+'存在未知项，请说明原因；静音也属于证据缺失。';
 }
 return null;
}
$('download').addEventListener('click',()=>{
 const error=validationError();if(error){$('error').textContent=error;return;}
 const blob=new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),link=document.createElement('a');
 link.href=url;link.download=data.assignment_id+'.ratings.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 dirty=false;$('status').textContent='已生成下载文件；已完成 '+data.items.filter(r=>r.completed).length+' / '+data.items.length+' 段。';
});
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
refresh();
</script></body></html>'''


def write_pages(directory):
    directory = Path(directory)
    outputs = []
    for path in sorted((directory/'assignments').glob('*.json')):
        form = read(path)
        page = assignment_page(form)
        output = directory/f"review_{form['assignment_id']}.html"
        if output.exists():
            raise ValueError(f'不覆盖已分发页面：{output}')
        outputs.append((output, page))
    if not outputs:
        raise ValueError('没有找到空白评审表')
    for output, page in outputs:
        output.write_text(page, encoding='utf-8')
    return len(outputs)


def probe(path):
    output = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return json.loads(output.stdout)


def from_timeline(args):
    timeline = read(args.timeline)
    media = probe(args.video)
    duration = float(media['format']['duration'])
    # This adapter targets a single-source timeline. Multi-source input must map
    # each source explicitly through the documented matches/sources JSON format.
    if any(e.get('source_id', 'single') != 'single' for e in timeline['events']):
        raise ValueError('多素材时间轴请使用显式 sources 映射，不能套用单素材适配器')
    events = []
    for e in timeline['events']:
        events.append({'event_id': e['event_id'], 'source_id': 'single',
                       'clip_start_sec': max(0, e['clip_start_sec']-args.context),
                       'clip_end_sec': min(duration, e['clip_end_sec']+args.context),
                       'sampling_origin': 'model_timeline_all_events'})
    return {'matches': [{'match_id': args.match_id, 'split': args.split,
        'pool_scope': 'candidate_pool_only', 'sources': [{'source_id': 'single',
            'path': str(args.video.resolve()), 'duration_sec': duration,
            'audio_available': any(s['codec_type'] == 'audio' for s in media['streams'])}],
        'events': events}]}


def render(directory):
    directory = Path(directory)
    private = read(directory/'private_mapping.json')
    verified = {}
    for item in private['items']:
        path = item['source_path']
        if path not in verified:
            if _sha_file(path) != item['source_sha256']:
                raise ValueError('原素材已改变，必须重新生成盲评包')
            info = probe(path)
            verified[path] = {'duration': float(info['format']['duration']),
                              'audio': any(s['codec_type'] == 'audio' for s in info['streams'])}
        if item['clip_end_sec'] > verified[path]['duration']+.04:
            raise ValueError('区间越过真实素材时长')
        if item['source_has_audio'] != verified[path]['audio']:
            raise ValueError('声明的音轨存在性与 ffprobe 不一致')
    manifest = {'benchmark_id': private['benchmark_id'], 'clips': []}
    (directory/'media').mkdir(exist_ok=True)
    for item in private['items']:
        for condition in ('audio', 'silent'):
            if condition == 'audio' and not item['source_has_audio']:
                continue
            target = directory/'media'/f"{item['item_id']}_{condition}.mp4"
            if target.exists():
                raise ValueError(f'为保护已分发素材，不覆盖现有文件：{target}')
            command = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(item['clip_start_sec']),
                       '-i', item['source_path'], '-t', str(item['duration_sec']),
                       '-map', '0:v:0', '-map_metadata', '-1', '-map_chapters', '-1',
                       '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p']
            command += ['-map', '0:a:0', '-c:a', 'aac', '-b:a', '192k'] if condition == 'audio' else ['-an']
            command += ['-movflags', '+faststart', str(target)]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                actual = probe(target)
                actual_duration = float(actual['format']['duration'])
                has_audio = any(s['codec_type'] == 'audio' for s in actual['streams'])
                if abs(actual_duration-item['duration_sec']) > .15 or has_audio != (condition == 'audio'):
                    raise ValueError('匿名片段时长或音轨校验失败')
            except Exception:
                target.unlink(missing_ok=True)
                raise
            manifest['clips'].append({'item_id': item['item_id'], 'condition': condition,
                                      'file': str(target.relative_to(directory)),
                                      'duration_sec': actual_duration, 'sha256': _sha_file(target)})
    write(directory/'media_manifest.json', manifest)
    return len(manifest['clips'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    timeline = sub.add_parser('import-timeline', help='将单素材模型时间轴转换成未标注候选池')
    timeline.add_argument('--timeline', type=Path, required=True)
    timeline.add_argument('--video', type=Path, required=True)
    timeline.add_argument('--match-id', required=True)
    timeline.add_argument('--split', choices=['train', 'dev', 'test'], required=True)
    timeline.add_argument('--context', type=float, default=3., help='前后额外上下文秒数，默认各 3 秒')
    timeline.add_argument('--output', type=Path, required=True)
    prep = sub.add_parser('prepare', help='仅生成全空评分及私有映射，不创建任何人工标签')
    prep.add_argument('--input', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--seed', type=int, default=20260912)
    prep.add_argument('--raters-per-condition', type=int, default=3)
    media = sub.add_parser('render', help='生成无模型文字/排名的匿名原速视频，静音副本移除音轨')
    media.add_argument('--package', type=Path, required=True)
    pages = sub.add_parser('pages', help='为现有空包生成每位评员的独立离线 HTML 页面')
    pages.add_argument('--package', type=Path, required=True)
    evaluate_parser = sub.add_parser('evaluate', help='导入声明由真人完成的评分；空表指标仍为未知')
    evaluate_parser.add_argument('--package', type=Path, required=True)
    evaluate_parser.add_argument('--ratings', type=Path, nargs='+', required=True)
    evaluate_parser.add_argument('--predictions', type=Path)
    evaluate_parser.add_argument('--split', choices=['train', 'dev', 'test'], default='test')
    evaluate_parser.add_argument('--output', type=Path, required=True)
    evaluate_parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    try:
        if args.command == 'import-timeline':
            if not 0 <= args.context <= 60:
                raise ValueError('--context 必须在 0–60 秒之间')
            write(args.output, from_timeline(args))
        elif args.command == 'prepare':
            if args.output.exists():
                raise ValueError('输出目录已存在；请使用新目录，避免覆盖已分发评分包')
            private, forms = prepare(read(args.input), args.seed, args.raters_per_condition)
            write(args.output/'private_mapping.json', private)
            for form in forms:
                write(args.output/'assignments'/f"{form['assignment_id']}.json", form)
            write_pages(args.output)
            print(json.dumps({'items': len(private['items']), 'assignments': len(forms),
                              'human_labels_created': 0}, ensure_ascii=False))
        elif args.command == 'render':
            print(json.dumps({'rendered_clips': render(args.package)}))
        elif args.command == 'pages':
            print(json.dumps({'offline_pages': write_pages(args.package)}))
        else:
            report = evaluate(read(args.package/'private_mapping.json'), [read(p) for p in args.ratings],
                              read(args.predictions) if args.predictions else None,
                              split=args.split, bootstrap_samples=args.bootstrap_samples)
            write(args.output, report)
            print(json.dumps({'human_completed_rows': report['human_completed_rows'],
                              'matches': report['matches']}, ensure_ascii=False))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
