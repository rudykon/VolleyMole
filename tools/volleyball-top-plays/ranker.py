import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from common import APP, save_json
from schemas import DECISION_SCHEMA, validate_decision
from semantic import candidate_fields, image_parts, request_json, choose_vision_model, visual_reviews


def shortlist(manifest, top_k):
    candidates=sorted((r for r in manifest['rallies'] if r['eligible']),key=lambda r:(-r['rule_score'],r['start_sec']))
    if len(candidates)<top_k:
        raise ValueError(f'只有 {len(candidates)} 个有效回合，不能生成 {top_k} 佳球；请检查回合证据或调整配置')
    return candidates[:max(top_k,manifest['config']['preview_limit'])]


def rule_decision(candidates, top_k):
    selected=[]
    pool=list(candidates)
    while pool and len(selected)<top_k:
        # Mild time/action diversity penalty; do not invent evidence for variety.
        def adjusted(r):
            penalty=sum(3 for s in selected if abs(s['start_sec']-r['start_sec'])<90)
            penalty+=sum(1 for s in selected if s['actions']==r['actions'])
            return r['rule_score']-penalty
        best=max(pool,key=adjusted); selected.append(best); pool.remove(best)
    result=[]
    titles=['多拍攻防','连续往返','防守与配合','耐心组织','攻防交锋']
    for rank,r in enumerate(selected,1):
        m=r['ball_metrics']
        result.append({'rank':rank,'rally_id':r['rally_id'],'clip_start_sec':r['safe_start_sec'],'clip_end_sec':r['safe_end_sec'],
                       'title':titles[(rank-1)%len(titles)],
                       'reason':f"持续 {r['duration_sec']:.1f} 秒，有效球轨迹 {m['visible_ratio']:.0%}，轨迹转折 {m['trajectory_changes']} 次；规则评分 {r['rule_score']:.1f}。动作与得分未经语义确认。",
                       'confidence':round(min(.85,.4+.4*m['visible_ratio']),3)})
    return {'title':f'比赛{top_k}佳球','selected':result}


def api_decision(candidates, directory, top_k, focus, endpoint, model, key, timeout, reviews=None, audit=None):
    prompt=(APP/'prompts/rank_top_plays.md').read_text(encoding='utf-8')
    content=[{'type':'text','text':f'选择 {top_k} 个回合。目标号码：{focus}。'}]
    for r in candidates:
        fields=candidate_fields(r)
        content.append({'type':'text','text':json.dumps(fields,ensure_ascii=False)})
        if reviews is None:
            content.extend(image_parts(r,directory))
        else:
            review=next(v for v in reviews if v['rally_id']==r['rally_id'])
            content.append({'type':'text','text':'视觉模型根据这三张关键帧给出的评审（保留不确定性）：'+json.dumps(review,ensure_ascii=False)})
    if reviews is not None:
        prompt+='\n本次由独立视觉模型查看截图，你接收到的是视觉证据 JSON；不要声称自己直接看过图片。'
    payload={'model':model,'messages':[{'role':'system','content':prompt},{'role':'user','content':content}],
             'response_format':{'type':'json_schema','json_schema':{'name':'volleyball_edit_decision','strict':True,'schema':DECISION_SCHEMA}}}
    result,metadata=request_json(endpoint,key,payload,timeout)
    if audit is not None:audit.append(metadata)
    if not isinstance(result,dict) or set(result)!={'title','selected'}:
        raise ValueError('模型剪辑单根字段不符合 schema')
    allowed={r['rally_id'] for r in candidates}
    if any(r['rally_id'] not in allowed for r in result['selected']):
        raise ValueError('模型返回预筛候选以外的回合')
    return result


def rank(manifest,directory,top_k,focus,mode,endpoint,model,timeout=90,vision_model=None):
    candidates=shortlist(manifest,top_k)
    key=os.getenv('VOLLEYMOLE_API_KEY') or os.getenv('OPENAI_API_KEY')
    failure=None
    decision=None
    semantic_files=[];requests=[];routing=None
    if mode!='rules' and key and model:
        try:
            if not vision_model:
                try:
                    decision=api_decision(candidates,directory,top_k,focus,endpoint,model,key,timeout,audit=requests)
                except HTTPError as exc:
                    body=exc.read().decode(errors='replace')
                    exc.close()
                    if exc.code!=400 or 'not a multimodal model' not in body.lower():raise
                    vision_model,advertised=choose_vision_model(endpoint,key,timeout)
                    routing={'reason':'primary_model_does_not_accept_images','primary_http_status':400,
                             'primary_model':model,'vision_model':vision_model,'advertised_models':advertised}
            if vision_model:
                reviews,semantic_files,batches=visual_reviews(candidates,directory,endpoint,vision_model,key,timeout)
                requests.extend(batches)
                decision=api_decision(candidates,directory,top_k,focus,endpoint,model,key,timeout,reviews=reviews,audit=requests)
            validate_decision(decision,manifest,directory,top_k)
        except (HTTPError,URLError,TimeoutError,ValueError,KeyError,TypeError,IndexError,OSError) as exc:
            decision=None
            # Exception bodies can contain remote data; record only a safe category.
            failure={'type':type(exc).__name__,'message':'语义排序请求失败或返回无效剪辑单，已按规则降级。'}
            if isinstance(exc,HTTPError):
                failure['http_status']=exc.code
                exc.close()
    else:
        failure={'type':'disabled' if mode=='rules' else 'not_configured','message':'使用规则排序；未调用语义模型。'}
    if decision is None:
        decision=rule_decision(candidates,top_k)
        ranking_mode='rules_fallback'
    else:
        ranking_mode='vision_then_text_api' if vision_model else 'multimodal_api'
    validate_decision(decision,manifest,directory,top_k)
    decision.update(ranking_mode=ranking_mode,model=model if ranking_mode!='rules_fallback' else None,
                    vision_model=vision_model if ranking_mode=='vision_then_text_api' else None,focus_player=focus,fallback=failure)
    selected={r['rally_id'] for r in decision['selected']}
    decision['rejected']=[{'rally_id':r['rally_id'],'rule_score':r['rule_score'],
                           'reasons':r['exclusion_reasons'] or ['综合排序未进入本次入选名额']} for r in manifest['rallies'] if r['rally_id'] not in selected]
    path=Path(directory)/'edit_decision.json'; save_json(path,decision)
    save_json(Path(directory)/'ranking_log.json',{'mode':ranking_mode,'candidate_ids':[r['rally_id'] for r in candidates],
              'fallback':failure,'routing':routing,'requests':requests,'max_frames_per_candidate':3,'uploaded_whole_video':False})
    return str(path),[path,Path(directory)/'ranking_log.json']+semantic_files
