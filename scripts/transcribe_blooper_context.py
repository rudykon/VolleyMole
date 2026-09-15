#!/usr/bin/env python3
"""Optional ASR context. Transcription never proves laughter or event success."""
import concurrent.futures,json,subprocess,sys,time,uuid
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from volleymole.common import read_json,save_json,digest

def main():
    directory=Path(sys.argv[1]);cfg=read_json(ROOT/'llm_api.json')['llm']
    data=read_json(directory/'evidence.json');out=directory/'asr_context';out.mkdir(exist_ok=True)
    def process(item):
        path=out/(item['id']+'.json')
        if path.exists():return read_json(path)
        result={'id':item['id'],'source_sha256':data['source_sha256'],'start_sec':item['start_sec'],
            'end_sec':item['end_sec'],'annotation_source':'whisper-1','is_ground_truth':False,
            'laughter_confirmed':None,'note':'ASR text is context only; may hallucinate, omit laughter or mishear speech.'}
        try:
            wav=subprocess.check_output(['ffmpeg','-v','error','-i',item['clip'],'-vn','-ac','1','-ar','16000','-f','wav','-'],timeout=20)
            boundary='VolleyMole'+uuid.uuid4().hex;parts=[]
            for key,value in [('model','whisper-1'),('language','zh'),('response_format','verbose_json')]:
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
            parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="context.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),wav,f'\r\n--{boundary}--\r\n'.encode()])
            request=Request(cfg['base_url'].rstrip('/')+'/audio/transcriptions',data=b''.join(parts),headers={
                'Authorization':'Bearer '+cfg['api_key'],'Content-Type':'multipart/form-data; boundary='+boundary})
            with urlopen(request,timeout=45) as response:raw=response.read(2*1024*1024)
            result.update(status='complete',response=json.loads(raw),waveform_bytes=len(wav))
        except HTTPError as exc:result.update(status='failed',http_status=exc.code);exc.close()
        except Exception as exc:result.update(status='failed',error=type(exc).__name__)
        save_json(path,result);print(item['id'],result['status'],flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:items=list(pool.map(process,data['items']))
    save_json(out/'report.json',{'items':items,'completed':sum(x['status']=='complete' for x in items),'script_sha256':digest(__file__)})

if __name__=='__main__':main()
