"""Customer task summaries from stage records and verification reports, never percentages."""
import os
from pathlib import Path
import re

GROUPS=[('analysis','分析比赛',{'inference','analytics','tracking','player','events'}),
        ('rallies','提取回合',{'rallies','previews'}),
        ('rank','选择高光',{'rank','ranking','replay_reviews'}),
        ('render','生成成片',{'render','render_lively'}),
        ('verify','检查输出',{'verify','verify_lively'})]


def summary(workspace,job,log):
    values=job.get('values',{})
    value=values.get('output') or values.get('render_output') or values.get('run')
    root=None
    try:
        if value:
            candidate=workspace.path(values.get('run') or value,exist=True)
            if candidate.is_dir() and candidate!=workspace.root: root=candidate
    except (ValueError,OSError): pass
    states=[]; verification=[]; visited=0
    started=job.get('started_at') or job.get('created_at',0)
    reused=set(re.findall(r'\[([\w-]+)\] 复用已验证结果',log))
    if root:
        for directory,dirs,files in os.walk(root,followlinks=False):
            folder=Path(directory);visited+=1
            if visited>500: break
            dirs[:]=[d for d in dirs if not d.startswith('.') and d not in {'frames','segments','previews','analytics','tracking','profiles','semantic_cache','media_cache','verification','verification_lively','ocr_runtime','event_frames'} and not (folder/d).is_symlink()]
            try: workspace.path(str(folder),exist=True)
            except (ValueError,OSError): dirs[:]=[];continue
            for name in ('state.json','verification.json','verification_lively.json'):
                path=folder/name
                if name not in files: continue
                try:
                    data=workspace.read_json(workspace.path(str(path),exist=True))
                    if not isinstance(data,dict): continue
                    if name=='state.json':
                        for key,entry in data.get('stages',{}).items():
                            if isinstance(entry,dict) and (entry.get('started_at',0)>=started-1 or key in reused):
                                states.append((key,entry.get('status','unknown')))
                    elif path.stat().st_mtime>=started-1 or name.removesuffix('.json').replace('verification','verify') in reused:
                        verification.append(data.get('status','unknown'))
                except (ValueError,OSError,TypeError,AttributeError): continue
    stages=[]
    for index,(key,label,names) in enumerate(GROUPS):
        observed=[status for name,status in states if name in names]
        status='unknown'
        if observed:
            status='failed' if 'failed' in observed else 'running' if 'running' in observed else 'complete' if all(s=='complete' for s in observed) else 'unknown'
            if status=='running' and job['status'] not in {'running','cancelling'}: status='interrupted'
            # A group may have more sub-stages or sets still to run. A completed
            # sub-stage alone does not prove the whole group has finished.
            later=set().union(*(group[2] for group in GROUPS[index+1:]))
            if status=='complete' and index<3 and job['status']!='succeeded' and not any(n in later for n,_ in states): status='partial'
        stages.append({'id':key,'label':label,'status':status})
    outputs=[]
    if job['status']=='succeeded':
        outputs=[item for item in workspace.library()['outputs'] if item['film_id'] in job.get('film_ids',[])]
    quality='unknown'
    if any(s in {'failed','error'} for s in verification) or stages[-1]['status']=='failed': quality='failed'
    elif outputs:
        from .films import detail
        checks=[detail(workspace,item['film_id'])['verification'] for item in outputs]
        quality='failed' if 'failed' in checks else 'passed' if all(s=='passed' for s in checks) else 'unknown'
    reason='任务尚未生成可播放成片。'
    if job['status']=='queued': reason='任务正在排队，开始处理后会显示阶段记录。'
    elif job['status'] in {'running','cancelling'}: reason='处理仍在进行，成片生成后可在这里播放。'
    elif not outputs: reason='本次任务未找到可播放成片；可能仅执行了分析、校验，或输出文件已移动。'
    if not outputs and job.get('command')=='meme-audio' and job['status']=='succeeded':
        reason='本次配音输出未关联已生成的成片，不归入“我的成片”；可在任务参数与分析报告中查看处理产物。'
    if job.get('publication_error'): reason=job['publication_error']
    return {'stages':stages,'outputs':outputs,'verification':quality,'message':reason,'has_stages':bool(states)}
