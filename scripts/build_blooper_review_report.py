#!/usr/bin/env python3
"""Package reviewed candidate findings, source evidence and original-audio clips."""
import argparse,html,json,subprocess,concurrent.futures
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volleymole.common import read_json,save_json,digest,probe

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
    evidence=read_json(root/'evidence.json');findings=read_json(root/'findings.json')
    by_id={i['id']:i for i in evidence['items']};ids=[i['id'] for i in findings['items']]
    if len(set(ids))!=len(ids) or set(ids)!=set(by_id):raise ValueError('Every original candidate must be accounted for exactly once')
    reserves=read_json(root/'reserve_manifest.json');cards=[];rows=[]
    def verify(item):
        path=Path(item['clip']);meta=probe(path)
        if digest(path)!=item['clip_sha256']:raise ValueError('Clip hash mismatch')
        if abs(meta['duration_sec']-(item['end_sec']-item['start_sec']))>.2 or not meta['has_audio']:raise ValueError('Clip context or original audio missing')
        subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],check=True,timeout=60)
        return {'id':item['id'],'duration_sec':meta['duration_sec'],'full_decode':'passed','original_audio_retained':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(verify,evidence['items']))
    for note in findings['items']:
        item=by_id[note['id']];frames=[{**f,'review_pass':'context_2fps'} for f in item['frames']]
        dense=root/item['id']/'dense.json'
        if dense.exists():frames += [{**f,'id':'dense_'+f['id'],'review_pass':'detail_8fps'} for f in read_json(dense)['evidence']]
        facts=[]
        for fact in note['facts']:
            ref=min(frames,key=lambda x:abs(x['start_sec']-fact['time_sec']))
            if abs(ref['start_sec']-fact['time_sec'])>.3:raise ValueError('Fact not grounded at supplied source time')
            facts.append({**fact,'evidence_id':ref['id'],'evidence_time_sec':ref['start_sec'],'review_pass':ref['review_pass']})
        peak=note['event_peak_sec']
        if peak is not None and not item['start_sec']<=peak<=item['end_sec']:raise ValueError('Event peak outside observed context')
        if note['suggested_clip']:
            lo,hi=note['suggested_clip']
            if not item['start_sec']<=lo<hi<=item['end_sec']:raise ValueError('Reserve clip exceeds reviewed context')
        response_file=root/'api_review'/(item['id']+'_retry.json')
        if not response_file.exists():response_file=root/'api_review'/(item['id']+'.json')
        api=read_json(response_file) if response_file.exists() else {'status':'not_run'}
        asr=read_json(root/'asr_context'/(item['id']+'.json'))
        row={**note,'facts':facts,'source_sha256':evidence['source_sha256'],'context_start_sec':item['start_sec'],
             'context_end_sec':item['end_sec'],'audio_peak_sec':item['center_sec'],
             'laughter_candidate_raw_probability':item['audio_candidate']['model_scores']['Laughter'],
             'related_laughter_score':None,'laughter_linked':None,'final_blooper_selected':False,
             'audio_peak_delay_sec':round(item['center_sec']-peak,3) if peak is not None else None,
             'external_visual_status':api['status'],'external_audio_asr_status':asr['status'],
             'external_visual_response_file':str(response_file),'asr_response_file':str(root/'asr_context'/(item['id']+'.json')),
             'clip':item['clip'],'evidence_frames':frames,'annotation_source':'model','is_ground_truth':False}
        rows.append(row)
        esc=html.escape;player=item['id'];target=(peak or item['center_sec'])-item['start_sec']
        facts_html=''.join(f'<li>{x["time_sec"]:.3f}秒：{esc(x["text"])}</li>' for x in facts)
        reserve_link=f'<a href="{player}/reserve.mp4" download>下载原声备选片段</a>' if note['decision']=='reserve' else ''
        cards.append(f'''<article data-state="{note['decision']}"><h2>{player[-2:]} · {esc(note['title'])}</h2>
<p><b>{esc(note['decision_label'])}</b> · 原片 {item['start_sec']:.2f}–{item['end_sec']:.2f} 秒</p>
<video id="{player}" controls preload="metadata" src="{player}/context.mp4"></video>
<button onclick="seek('{player}',{target:.3f})">定位动作 / 声音候选</button> {reserve_link}
<p>{esc(note['reason'])}</p><ul>{facts_html}</ul><p class="muted">{esc(note['uncertainty'])}</p>
<details><summary>证据与接口记录</summary><p>声音候选：{item['center_sec']:.2f}秒；未校准原始分数 {row['laughter_candidate_raw_probability']:.6f}。该分数不等于确认笑声。</p>
<p><a href="{player}/sheet_1.jpg">前半段时间帧</a> · <a href="{player}/sheet_2.jpg">后半段时间帧</a></p>
<p>视觉接口：{esc(api['status'])}；音频转写：{esc(asr['status'])}。转写与模型描述不是独立真值。</p></details></article>''')
    summary={'original_candidates':len(rows),'visually_reviewed':len(rows),'reserve_candidates':sum(r['decision']=='reserve' for r in rows),
        'not_selected':sum(r['decision']=='not_selected' for r in rows),'confirmed_laughter_associations':0,'final_bloopers':0,
        'external_visual_completed':sum(r['external_visual_status']=='complete' for r in rows),
        'external_audio_asr_completed':sum(r['external_audio_asr_status']=='complete' for r in rows),
        'context_frames':sum(len(i['frames']) for i in evidence['items']),
        'extra_detail_frames':sum(len(r['evidence_frames']) for r in rows)-sum(len(i['frames']) for i in evidence['items']),
        'user_authorized_local_upload':True,'schema_version':'volleymole-local-blooper-review-v1','is_ground_truth':False,
        'policy':'Laughter-first; ordinary serve/attack errors can qualify with linked laughter. Source audio peak alone cannot confirm the link.',
        'verification':checks,'reserved_video':reserves,'items':rows,
        'limitations':['Vision API missed visible ball motion and a fall in some clips; its prose was not accepted as ground truth.',
            'Vision model rejected actual audio input with HTTP 400; ASR is fallible speech context, not laughter confirmation.',
            'No confirmed serve-error or attack-error label is claimed when final contact is occluded.']}
    save_json(root/'report.json',summary)
    (root/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>8处声音候选复核</title><style>body{font:16px/1.65 system-ui;background:#f4f6f8;color:#18222c;max-width:1100px;margin:auto;padding:24px}article,header{background:white;padding:24px;margin:18px 0;border-radius:12px}video{width:100%;max-height:560px;background:#111}h1{font-size:28px}h2{font-size:22px}.muted{color:#5a6573}button,a{margin-right:14px}button{padding:8px;cursor:pointer}details{border-top:1px solid #ddd;padding-top:12px}</style>
<header><h1>8处声音候选复核</h1><p>3处失败处理备选 · 5处暂不入选 · 0处已确认笑声关联</p>
<p>全部为模型复核。原声保留在播放器中。备选没有被当作已确认的五大囧；原始声音分数、转写和视觉模型描述分别保存。</p>
<p><a href="失误备选_未确认笑声关联.mp4" download>下载3段原声备选合集</a><a href="report.json">完整结果 JSON</a></p>
<button onclick="filter('all')">全部8处</button><button onclick="filter('reserve')">3处备选</button></header>'''+''.join(cards)+'''
<script>function seek(id,t){let v=document.getElementById(id);v.currentTime=t;v.play()}function filter(s){document.querySelectorAll('article').forEach(e=>e.hidden=s!=='all'&&e.dataset.state!==s)}</script></html>''',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('original_candidates','reserve_candidates','not_selected','external_visual_completed','external_audio_asr_completed','context_frames','extra_detail_frames')}))

if __name__=='__main__':main()
