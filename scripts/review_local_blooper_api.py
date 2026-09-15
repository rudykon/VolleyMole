#!/usr/bin/env python3
"""Bounded review of explicitly authorized local candidate footage."""
import argparse,base64,concurrent.futures,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volleymole.common import read_json,save_json,digest
from volleymole.semantic import sampled_evidence,request_json
from urllib.error import HTTPError

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    cfg=read_json(ROOT/'llm_api.json')['llm'];source=read_json(a.directory/'evidence.json')
    items=source['items'];out=a.directory/'api_review';out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Do not overwrite prior review')
    began=time.monotonic();deadline=began+150;model='qwen3-vl-8b-instruct'
    def call(content,prompt):
        return request_json(cfg['base_url'],cfg['api_key'],{'model':model,'temperature':0,'max_tokens':1024,
            'messages':[{'role':'system','content':prompt},{'role':'user','content':content}],
            'response_format':{'type':'json_object'}},min(45,max(.01,deadline-time.monotonic())))
    # Audio capability is a separate, explicit real-waveform probe; accepting
    # a model name or an image request never proves audio support.
    item=items[0];wav=subprocess.check_output(['ffmpeg','-v','error','-ss',str(item['center_sec']-3),
        '-i',item['source']['path'],'-t','6','-vn','-ac','1','-ar','16000','-f','wav','-'],timeout=30)
    audio={'status':'unknown','source_start_sec':item['center_sec']-3,'duration_sec':6,'model':model}
    try:
        raw,meta=call([{'type':'input_audio','input_audio':{'data':base64.b64encode(wav).decode(),'format':'wav'}}],
            '只根据实际输入音频，用中文JSON说明你能否听到音频、听到哪些声音、有无笑声、可辨的原话。不能听到则明确未知，不猜测。')
        audio.update(status='response_received_capability_not_independently_verified',response=raw,request=meta)
    except HTTPError as exc:audio.update(status='unsupported_or_rejected',http_status=exc.code);exc.close()
    except Exception as exc:audio.update(status='failed',error=type(exc).__name__)
    save_json(out/'audio_capability.json',audio)
    def review(item):
        path=out/(item['id']+'.json')
        if path.exists():return read_json(path)
        result={'id':item['id'],'annotation_source':'model','is_ground_truth':False,'model':model,
            'source_sha256':source['source_sha256'],'start_sec':item['start_sec'],'end_sec':item['end_sec'],
            'audio_used':False,'sampling_fps':2,'width':224}
        try:
            evidence,content,_=sampled_evidence(item['source'],item['start_sec'],item['end_sec'],2,224,None,deadline)
            prompt='你是比赛录像描述器。请用中文JSON客观描述实际可见内容，普通等待和普通回合也必须描述，不要求好笑，不要求返回囧事。输出 scene（站位与状态）、actions（动作过程）、outcome（后果）、error_type（发球失误/扣球失误/接球失败/未见失误/未知）、uncertainty。只引用所给时间，不补出没有看到的触球或得分。你没有音频，不推断笑声。这是自动描述，不是真值。'
            response,metadata=call(content,prompt)
            result.update(status='complete',response=response,request=metadata,evidence=evidence)
        except HTTPError as exc:result.update(status='failed',error='HTTPError',http_status=exc.code);exc.close()
        except Exception as exc:result.update(status='failed',error=type(exc).__name__)
        save_json(path,result);print(item['id'],result['status'],flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(review,items))
    report={'model':model,'items':results,'audio_capability':audio,'elapsed_sec':time.monotonic()-began,
        'completed':sum(x['status']=='complete' for x in results),'requested':len(items),
        'script_sha256':digest(__file__),'endpoint':cfg['base_url'],'user_authorized_local_upload':True}
    save_json(out/'report.json',report);print(json.dumps({k:report[k] for k in ('completed','requested','elapsed_sec')}))

if __name__=='__main__':main()
