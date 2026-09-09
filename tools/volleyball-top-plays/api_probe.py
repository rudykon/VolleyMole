"""Check configured vision/schema support with one local preview, without logging keys."""
import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--model')
    args=p.parse_args();config=json.loads(args.config.read_text())['llm']
    schema={'type':'object','properties':{'description':{'type':'string'}},'required':['description'],'additionalProperties':False}
    payload={'model':args.model or config['model'],'messages':[{'role':'user','content':[
        {'type':'text','text':'用一句中文描述这张比赛截图，返回 JSON description 字段。'},
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(args.image.read_bytes()).decode(),'detail':'low'}}]}],
        'response_format':{'type':'json_schema','json_schema':{'name':'frame_description','strict':True,'schema':schema}}}
    req=Request(config['base_url'].rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={
        'Content-Type':'application/json','Authorization':'Bearer '+config['api_key']})
    try:
        with urlopen(req,timeout=120) as response:
            data=json.load(response)
            print(json.dumps({'status':response.status,'choices':data.get('choices'),'usage':data.get('usage')},ensure_ascii=False))
    except HTTPError as exc:
        body=exc.read().decode(errors='replace').replace(config['api_key'],'[REDACTED]')
        print('HTTP',exc.code,body[:1800])
