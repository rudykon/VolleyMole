"""Local VolleyMole workspace UI, using only Python's standard HTTP stack."""
import argparse
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit
import uuid

from .catalog import catalog, build_argv
from .jobs import JobManager, redact
from .media import MediaCache
from .library import collections as media_collections
from .progress import summary as task_summary
from . import films

STATIC=Path(__file__).with_name('static')
MEDIA={'.mp4','.mov','.mkv','.avi','.webm','.m4v','.mts','.m2ts','.wav','.mp3','.m4a','.aac','.flac','.ogg'}
VIDEO={'.mp4','.mov','.mkv','.avi','.webm','.m4v','.mts','.m2ts'}
DOWNLOAD=MEDIA|{'.json','.jsonl','.csv','.txt','.log','.png','.jpg','.jpeg','.webp','.zip','.md','.pdf'}
PROTECTED={'src','scripts','tests','docs','build','dist','refer','venv','__pycache__'}
SECRET=re.compile(r'(^\.|api.*\.json$|token|secret|credential|password|\.pem$|\.key$)',re.I)
MAX_JSON=2*1024*1024


def listen_host(value):
    if value=='localhost': return value
    try:
        address=ipaddress.IPv4Address(value)
        if address.is_unspecified or address.is_multicast or value=='255.255.255.255':
            raise ipaddress.AddressValueError()
        return str(address)
    except ipaddress.AddressValueError:
        raise argparse.ArgumentTypeError('请指定具体的本机 IPv4 地址或 localhost；可重复 --host 监听多个地址') from None


class Workspace:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.token=secrets.token_urlsafe(32)
        self.jobs=JobManager(self.root)
        self.settings_lock=threading.RLock()
        self.media_cache=MediaCache(self)

    def path(self,value,*,exist=False):
        if not isinstance(value,str) or not value or '\x00' in value: raise ValueError('请选择工作区内的路径')
        path=Path(value).expanduser()
        path=(self.root/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root): raise ValueError('只能访问当前工作区内的文件')
        parts=path.relative_to(self.root).parts
        if any(SECRET.search(p) for p in parts if p!='.analysis-cache') or (parts and parts[0] in PROTECTED):
            # The one credential file is accepted as an argument, never served.
            raise ValueError('此路径属于程序或私密文件，不能通过文件工作台访问')
        if exist and not path.exists(): raise ValueError('文件或目录不存在')
        return path

    def argument_path(self,value,key=None):
        p=Path(value)
        if key=='llm_config' and (self.root/p).resolve()==self.root/'llm_api.json': return self.root/'llm_api.json'
        # Font paths are a read-only CLI input; the system font directory is safe.
        if key=='font' and p.is_absolute() and p.resolve().is_relative_to('/usr/share/fonts') and p.is_file(): return p.resolve()
        result=self.path(value)
        if key in {'output','render_output','directory','run','analysis_cache_dir'}:
            if result.is_relative_to(self.root/films.SHELF):
                raise ValueError('成片目录由收录流程管理，请将后台输出放在 runs 下')
            relative=result.relative_to(self.root)
            if not relative.parts or (len(relative.parts)==1 and result.is_file()):
                raise ValueError('输出或工作目录不能指向项目根目录及其已有文件')
        return result

    def relative(self,path):
        return str(Path(path).relative_to(self.root))

    def read_json(self,path):
        if path.stat().st_size>MAX_JSON: raise ValueError('JSON 文件过大，请下载后查看')
        return json.loads(path.read_text(encoding='utf-8'))

    def files(self,value='.',offset=0):
        p=self.path(value,exist=True)
        if not p.is_dir(): raise ValueError('请选择目录')
        entries=[]
        for child in p.iterdir():
            try: self.path(str(child),exist=True)
            except (ValueError,OSError): continue
            if not child.is_dir() and child.suffix.lower() not in DOWNLOAD|{'.pt','.pth','.onnx','.ttf','.otf'}: continue
            entries.append(child)
        entries.sort(key=lambda x:(not x.is_dir(),x.name.lower()))
        return {'path':self.relative(p),'parent':self.relative(p.parent) if p!=self.root else None,'total':len(entries),
                'entries':[{'name':x.name,'path':self.relative(x),'directory':x.is_dir(),
                            'size':0 if x.is_dir() else x.stat().st_size,'modified':x.stat().st_mtime}
                           for x in entries[offset:offset+200]],'offset':offset}

    def templates(self):
        from ..templates import BUILTINS, read_template
        result=[]
        for root,builtin in ((BUILTINS,True),(self.root/'templates',False)):
            for p in sorted(root.glob('*.json')):
                try: result.append({**read_template(p),'builtin':builtin})
                except (ValueError,OSError): continue
        return result

    def resources(self):
        from ..models import ModelRegistry
        from ..assets import asset_root,load_manifest
        registry=ModelRegistry()
        models=[{'name':name,'bytes':e['bytes'],'present':registry.target(name).is_file(),
                 'size_matches':registry.target(name).is_file() and registry.target(name).stat().st_size==e['bytes']}
                for name,e in registry.entries.items()]
        manifest=load_manifest();root=asset_root()
        if not os.environ.get('VOLLEYMOLE_ASSETS') and (self.root/'data/visual-assets').is_dir():
            root=self.root/'data/visual-assets'
        present=sum((root/name).is_file() for name in manifest['files'])
        return {'models':models,'model_directory':str(registry.directory),
                'assets':{'directory':str(root),'present':present,'total':len(manifest['files']),'version':manifest['version']},
                'ffmpeg':bool(shutil.which('ffmpeg')),'ffprobe':bool(shutil.which('ffprobe')),
                'python':sys.version.split()[0],'disk_free':shutil.disk_usage(self.root).free}

    def runs(self):
        rows=[]
        for base in ('runs','outputs'):
            for root,dirs,files in os.walk(self.root/base,followlinks=False):
                dirs[:]=[d for d in dirs if not d.startswith('.') and d not in {'analytics','tracking','previews','profiles','frames','segments','semantic_cache','media_cache','verification','verification_lively','ocr_runtime'}]
                if not {'run_config.json','state.json','edit_decision.json','render_report_lively.json','render_report.json','matches_report.json'}.intersection(files): continue
                p=Path(root)
                try: self.path(str(p))
                except ValueError: continue
                rows.append({'path':self.relative(p),'name':p.name,'modified':max((p/f).stat().st_mtime for f in files),
                             'rendered':any(f.startswith('render_report') for f in files),
                             'has_decision':'edit_decision.json' in files})
                if len(rows)>=400: break
        return sorted(rows,key=lambda r:r['modified'],reverse=True)

    def library(self):
        return media_collections(self)

    def media_path(self, value):
        path = self.path(value, exist=True)
        relative = path.relative_to(self.root)
        from .library import INTERNAL
        if relative.is_relative_to('data') and not INTERNAL.intersection(relative.parts):
            return path
        if path.parent.parent == self.root / films.SHELF and path.name in {'video.mp4', 'poster.jpg'}:
            films.detail(self, path.parent.name)
            return path
        raise PermissionError('后台产物不能直接展示，请先收录为成片')

    def run_detail(self,value):
        p=self.path(value,exist=True)
        if not p.is_dir(): raise ValueError('请选择运行目录')
        reports={}
        for name in ('state.json','run_config.json','edit_decision.json','render_report.json','render_report_lively.json','replay_reviews.json','verification.json','verification_lively.json','alignment_verification_lively.json','matches_report.json','audio_plan.json','events_report.json','collections_report.json','event_timeline.json','audio_events.json','timing_latest.json','title_card_readability.json'):
            f=p/name
            if f.is_file():
                try: reports[name]=redact(self.read_json(self.path(str(f))))
                except (ValueError,OSError): reports[name]={'note':'文件暂不可读或过大，请在文件列表中下载'}
        artifacts=[]
        for root,dirs,files in os.walk(p,followlinks=False):
            dirs[:]=[d for d in dirs if not d.startswith('.') and d not in {'analytics','tracking','profiles','frames','semantic_cache','ocr_runtime'}]
            for name in files:
                f=Path(root)/name
                if f.suffix.lower() not in MEDIA|{'.json','.csv'}: continue
                try: self.path(str(f))
                except ValueError: continue
                artifacts.append({'name':str(f.relative_to(p)),'path':self.relative(f),'size':f.stat().st_size})
                if len(artifacts)>=500: break
            if len(artifacts)>=500: break
        return {'path':self.relative(p),'reports':reports,'artifacts':artifacts,'limited':len(artifacts)>=500}

    def resume(self,value):
        from .catalog import fields,parser_for
        p=self.path(value,exist=True)
        config=self.read_json(self.path(str(p/'run_config.json'),exist=True))
        mode='match' if config.get('mode')=='date_grouped_match' else 'run'
        allowed={f['key'] for f in fields(parser_for(mode))}
        values={k:v for k,v in config.items() if k in allowed and v is not None and not isinstance(v,(dict,list))}
        values.update({k:v for k,v in config.get('performance',{}).items() if k in allowed and v is not None})
        if config.get('devices'): values['devices']=','.join(config['devices'])
        values.update(output=str(p.parent if mode=='match' else p),rerun_from='render')
        # Capture template options directly: the named user template may have moved.
        template=config.get('template')
        if isinstance(template,dict):
            for k,v in template.get('options',{}).items():
                if k in allowed and k not in values: values[k]=v
        if mode=='match':
            manifest=self.read_json(self.path(str(p/'match_manifest.json'),exist=True))
            parents={str(self.argument_path(s['path']).parent) for s in manifest.get('sources',{}).values()}
            if len(parents)!=1: raise ValueError('无法还原同一录像目录，请手动选择多局输入目录')
            values['input_dir']=parents.pop()
            values['date']=config['date']
        elif config.get('config'):
            # Runtime config is nested in run_config; persist a separate, portable input.
            saved=self.path('data/web-config/resume-'+uuid.uuid4().hex[:12]+'.json')
            saved.parent.mkdir(parents=True,exist_ok=True)
            saved.write_text(json.dumps(config['config'],ensure_ascii=False,indent=2))
            values['config']=str(saved)
        # Ranking mode is recorded with the decision, not in older run configs.
        decision=p/'edit_decision.json'
        if decision.is_file():
            ranking=self.read_json(decision).get('ranking_mode','')
            values['ranker']='rules' if ranking.startswith('rules') else 'auto'
        return {'command':mode,'values':values}

    def settings(self):
        with self.settings_lock:
            p=self.root/'llm_api.json'
            doc=self.read_json(p) if p.exists() else {}
            llm=doc.get('llm',{})
            return {k:llm.get(k,'') for k in ('base_url','model','vision_model')}|{'has_key':bool(llm.get('api_key'))}

    def save_settings(self,body):
        with self.settings_lock:
            p=self.root/'llm_api.json'
            doc=self.read_json(p) if p.exists() else {}
            llm=doc.setdefault('llm',{})
            url=body.get('base_url','').strip()
            parsed=urlsplit(url)
            if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('API 地址须为完整的 HTTP / HTTPS 地址，不能包含密钥')
            if not isinstance(body.get('model'),str) or not body['model'].strip(): raise ValueError('请填写模型名称')
            llm.update(provider='openai_compatible',base_url=url,model=body['model'].strip(),vision_model=str(body.get('vision_model','')).strip())
            if body.get('clear_key'): llm.pop('api_key',None)
            elif body.get('api_key'):
                if not isinstance(body['api_key'],str): raise ValueError('密钥必须是文本')
                llm['api_key']=body['api_key'].strip()
            temp=p.with_name('.llm_api.web.tmp')
            fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(fd,'w') as f: json.dump(doc,f,ensure_ascii=False,indent=2)
            os.chmod(temp,0o600);temp.replace(p)
            return self.settings()


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,workspace):
        self.workspace=workspace
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    server_version='VolleyMole'
    def log_message(self,fmt,*args): pass

    @property
    def app(self): return self.server.workspace

    def send_json(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        if self.command!='HEAD': self.wfile.write(data)

    def guard(self,mutate=False):
        host=self.headers.get('Host','')
        # The accepted socket identifies the real interface used by this request.
        # Additional listeners must not turn Host validation into a wildcard.
        local_address=self.connection.getsockname()[0]
        hosts={'localhost','127.0.0.1',local_address}
        allowed={f'{h}:{self.server.server_port}' for h in hosts}
        if self.server.server_port==80: allowed.update(hosts)
        if host not in allowed: raise PermissionError('不接受此 Host；请使用本机地址或当前连接的服务器 IP')
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://{host}': raise PermissionError('不接受跨站请求')
        if mutate and not secrets.compare_digest(self.headers.get('X-VolleyMole-Token',''),self.app.token):
            raise PermissionError('页面会话已失效，请刷新后重试')

    def body(self):
        n=int(self.headers.get('Content-Length','0'))
        if not 0<n<=MAX_JSON: raise ValueError('请求体为空或过大')
        value=json.loads(self.rfile.read(n))
        if not isinstance(value,dict): raise ValueError('请求体必须是 JSON 对象')
        return value

    def do_HEAD(self): self.do_GET()

    def do_GET(self):
        try:
            self.guard()
            url=urlsplit(self.path);q=parse_qs(url.query);get=lambda k,d='':q.get(k,[d])[0]
            if url.path=='/api/bootstrap':
                return self.send_json({'token':self.app.token,'workspace':str(self.app.root),'commands':catalog(),
                                       'templates':self.app.templates(),'resources':self.app.resources()})
            if url.path=='/api/files': return self.send_json(self.app.files(get('path','.'),max(0,int(get('offset','0')))))
            if url.path=='/api/jobs': return self.send_json(self.app.jobs.list())
            if url.path.startswith('/api/jobs/'):
                job_id=url.path.rsplit('/',1)[-1];job=self.app.jobs.get(job_id)
                log=self.app.jobs.log(job_id)
                return self.send_json({**job,'log':log,'progress':task_summary(self.app,job,log)})
            if url.path=='/api/runs': return self.send_json(self.app.runs())
            if url.path=='/api/library': return self.send_json(self.app.library())
            if url.path=='/api/film': return self.send_json(films.detail(self.app,get('id')))
            if url.path=='/api/media-info':
                self.app.media_path(get('path'))
                return self.send_json(films.media_info(self.app,get('path')) or self.app.media_cache.info(get('path')))
            if url.path=='/api/thumbnail': return self.serve_file(self.app.media_cache.thumbnail(get('path')))
            if url.path=='/api/run': return self.send_json(self.app.run_detail(get('path')))
            if url.path=='/api/templates': return self.send_json(self.app.templates())
            if url.path=='/api/settings': return self.send_json(self.app.settings())
            if url.path=='/api/resources': return self.send_json(self.app.resources())
            if url.path=='/api/defaults':
                return self.send_json(json.loads(Path(__file__).parents[1].joinpath('defaults.json').read_text()))
            if url.path=='/api/json': return self.send_json(redact(self.app.read_json(self.app.path(get('path'),exist=True))))
            if url.path=='/api/matches':
                from ..match_collection import discover_matches
                groups=discover_matches(self.app.path(get('path'),exist=True))
                return self.send_json([{'date':day,'sets':[{'number':n,'path':self.app.relative(p)} for n,p in sets]} for day,sets in groups.items()])
            if url.path=='/api/probe':
                p=self.app.path(get('path'),exist=True)
                if p.suffix.lower() not in MEDIA: raise ValueError('请选择音视频文件')
                try:
                    result=subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(p)],capture_output=True,text=True,timeout=20)
                    if result.returncode: raise ValueError('无法识别音视频文件')
                except FileNotFoundError: raise ValueError('未安装 FFprobe，请先安装并加入 PATH') from None
                return self.send_json(json.loads(result.stdout))
            if url.path=='/media':
                p=self.app.path(get('path'),exist=True)
                if p.suffix.lower() not in DOWNLOAD: raise ValueError('不支持此文件格式')
                if p.suffix.lower() in MEDIA: p=self.app.media_path(get('path'))
                if p.suffix.lower()=='.json':
                    # Reports can contain copied configuration. Redact before download.
                    data=json.dumps(redact(self.app.read_json(p)),ensure_ascii=False,indent=2).encode()
                    return self.serve_bytes(data,'application/json',p.name if get('download') else None)
                if p.suffix.lower() in {'.log','.txt','.jsonl','.csv','.md'}:
                    if p.stat().st_size>10*1024*1024: raise ValueError('文本超过网页安全预览大小，请在服务器本地读取')
                    return self.serve_bytes(redact(p.read_text(errors='replace')).encode(),'text/plain; charset=utf-8',p.name if get('download') else None)
                return self.serve_file(p,download=bool(get('download')))
            if url.path=='/': return self.serve_file(STATIC/'index.html')
            if url.path.startswith('/static/'):
                p=(STATIC/unquote(url.path[len('/static/'):])).resolve()
                if not p.is_relative_to(STATIC.resolve()): raise PermissionError('非法静态资源路径')
                return self.serve_file(p)
            return self.send_json({'error':'页面不存在'},404)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc: self.send_json({'error':str(exc)},400)
        except PermissionError as exc: self.send_json({'error':str(exc)},403)
        except FileNotFoundError: self.send_json({'error':'文件不存在'},404)
        except (BrokenPipeError,ConnectionResetError): pass
        except Exception as exc: self.send_json({'error':redact(str(exc))},500)

    def serve_bytes(self,data,kind,download=None):
        self.send_response(200);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        if download: self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+quote(download))
        self.end_headers()
        if self.command!='HEAD': self.wfile.write(data)

    def serve_file(self,path,download=False):
        if not path.is_file(): raise FileNotFoundError(path)
        size=path.stat().st_size;start=0;end=size-1;status=200
        value=self.headers.get('Range')
        if value:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
            if not match or not any(match.groups()): return self.range_error(size)
            first,last=match.groups()
            if not first:
                if int(last)==0: return self.range_error(size)
                start=max(0,size-int(last))
            else:
                start=int(first);end=min(size-1,int(last)) if last else size-1
            if start>=size or start>end: return self.range_error(size)
            status=206
        with path.open('rb') as f:
            self.send_response(status)
            self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(max(0,end-start+1)))
            self.send_header('Accept-Ranges','bytes');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-cache')
            if path.suffix=='.html':
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'")
            if status==206: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            if download: self.send_header('Content-Disposition',"attachment; filename*=UTF-8''"+quote(path.name))
            self.end_headers()
            if self.command=='HEAD': return
            f.seek(start);remaining=end-start+1
            while remaining>0:
                chunk=f.read(min(1024*1024,remaining))
                if not chunk: break
                self.wfile.write(chunk);remaining-=len(chunk)

    def range_error(self,size):
        self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
        self.send_header('Content-Length','0');self.end_headers()

    def do_POST(self):
        try:
            self.guard(mutate=True)
            url=urlsplit(self.path);q=parse_qs(url.query)
            if url.path=='/api/upload': return self.upload(q)
            body=self.body()
            if url.path=='/api/publish':
                if any(j['status'] in {'queued','running','cancelling'} for j in self.app.jobs.list()):
                    raise ValueError('请等待当前任务结束后收录已有成片')
                ids=films.publish_run(self.app,body.get('run'),title=body.get('title',''))
                if not ids: raise ValueError('未找到可收录的最终成片，请确认交付视频仍然存在')
                return self.send_json({'film_ids':ids},201)
            if url.path=='/api/resume': return self.send_json(self.app.resume(body.get('path')))
            if url.path=='/api/jobs':
                command=body.get('command');sub=body.get('subcommand');values=body.get('values',{}).copy()
                if command in ('run','match','infer') and not values.get('output'):
                    values['output']='runs/web/'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
                # Web resource installation stays in this workspace by default.
                if command=='models' and not values.get('directory'): values['directory']='models'
                if command=='assets' and sub!='verify' and not values.get('directory'): values['directory']='data/visual-assets'
                argv,clean=build_argv(command,sub,values,self.app.argument_path)
                return self.send_json(self.app.jobs.submit(command,sub,argv,clean,body.get('title','')),201)
            if url.path.startswith('/api/cancel/'):
                return self.send_json(self.app.jobs.cancel(url.path.rsplit('/',1)[-1]))
            if url.path=='/api/templates':
                from ..templates import BUILTINS,write_template,validate
                doc=validate(body)
                if (BUILTINS/(doc['name']+'.json')).exists(): raise ValueError('内置模板不可覆盖，请修改名称后另存')
                path=self.app.path('templates/'+doc['name']+'.json')
                write_template(doc,path)
                return self.send_json({'path':self.app.relative(path)},201)
            if url.path=='/api/settings': return self.send_json(self.app.save_settings(body))
            if url.path=='/api/document':
                kind=body.get('kind');doc=body.get('document')
                if kind=='audio-plan':
                    from ..meme_audio import validate_plan,media_info
                    video=self.app.path(body.get('video'),exist=True)
                    data=media_info(video)
                    duration=max(float(s.get('duration',0)) for s in data['streams'])
                    if not isinstance(doc,dict) or not isinstance(doc.get('cues'),list): raise ValueError('配音计划格式错误')
                    for cue in doc['cues']: cue['asset']=str(self.app.path(cue.get('asset'),exist=True))
                    validate_plan(doc,self.app.root,duration)
                elif kind=='algorithm':
                    defaults=json.loads(Path(__file__).parents[1].joinpath('defaults.json').read_text())
                    def check(d,reference):
                        if not isinstance(d,dict) or set(d)-set(reference): raise ValueError('算法配置含未知字段')
                        for k,v in d.items():
                            if isinstance(reference[k],dict): check(v,reference[k])
                            elif isinstance(reference[k],str):
                                if not isinstance(v,str): raise ValueError('配置类型错误：'+k)
                            elif type(v) not in (float,int) or not __import__('math').isfinite(v): raise ValueError('配置数值无效：'+k)
                    check(doc,defaults)
                else: raise ValueError('不支持此文档类型')
                p=self.app.path('data/web-config/'+kind+'-'+uuid.uuid4().hex[:12]+'.json')
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('x') as f: json.dump(doc,f,ensure_ascii=False,indent=2,allow_nan=False)
                return self.send_json({'path':self.app.relative(p)},201)
            return self.send_json({'error':'接口不存在'},404)
        except (ValueError,KeyError,TypeError) as exc: self.send_json({'error':str(exc)},400)
        except FileExistsError: self.send_json({'error':'同名文件已存在，请使用新名称'},409)
        except PermissionError as exc: self.send_json({'error':str(exc)},403)
        except (BrokenPipeError,ConnectionResetError): pass
        except Exception as exc: self.send_json({'error':redact(str(exc))},500)

    def upload(self,q):
        name=q.get('name',[''])[0];batch=q.get('batch',[uuid.uuid4().hex])[0]
        if not re.fullmatch(r'[a-zA-Z0-9-]{1,64}',batch): raise ValueError('上传批次无效')
        if not name or Path(name).name!=name or '/' in name or '\\' in name or SECRET.search(name): raise ValueError('文件名无效')
        if Path(name).suffix.lower() not in MEDIA|{'.json','.zip','.pt','.pth','.onnx','.ttf','.otf'}: raise ValueError('不支持此文件类型')
        n=int(self.headers.get('Content-Length','0'))
        if not 0<n<=50*1024**3: raise ValueError('单文件大小须在 0–50 GB 之间')
        purpose=q.get('purpose',[''])[0]
        if purpose not in {'','sources','materials'}: raise ValueError('上传用途无效')
        if purpose=='sources' and Path(name).suffix.lower() not in VIDEO: raise ValueError('比赛源片只接受视频文件')
        if purpose=='materials' and Path(name).suffix.lower() not in MEDIA: raise ValueError('剪辑素材只接受视频或音频文件')
        location='data/uploads/'+(purpose+'/' if purpose else '')+batch+'/'+name
        target=self.app.path(location)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            remaining=n
            with temporary.open('xb') as f:
                while remaining:
                    chunk=self.rfile.read(min(4*1024*1024,remaining))
                    if not chunk: raise ValueError('上传中断，请重新上传')
                    f.write(chunk);remaining-=len(chunk)
            # Atomic and no overwrite, including concurrent uploads of the same name.
            os.link(temporary,target)
        finally: temporary.unlink(missing_ok=True)
        self.send_json({'path':self.app.relative(target),'directory':self.app.relative(target.parent),'size':n},201)


def main(argv=None):
    parser=argparse.ArgumentParser(description='VolleyMole 中文网页工作台（本机 / 局域网 / SSH 转发）')
    parser.add_argument('--host',type=listen_host,action='append',
                        help='监听指定 IPv4 地址；可重复指定，默认仅 127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--workspace',type=Path,default=Path.cwd())
    parser.add_argument('--open',action='store_true',help='启动后打开系统浏览器')
    args=parser.parse_args(argv)
    root=args.workspace.expanduser().resolve()
    if not root.is_dir(): parser.error('工作区目录不存在')
    os.chdir(root);os.environ['VOLLEYMOLE_WORKSPACE']=str(root)
    os.environ['VOLLEYMOLE_TEMPLATES']=str(root/'templates')
    if (root/'data/visual-assets').is_dir(): os.environ.setdefault('VOLLEYMOLE_ASSETS',str(root/'data/visual-assets'))
    app=Workspace(root)
    servers=[]
    background=[]
    try:
        hosts=list(dict.fromkeys('127.0.0.1' if h=='localhost' else h for h in (args.host or ['127.0.0.1'])))
        for host in hosts: servers.append(Server((host,args.port),app))
        for server in servers[1:]:
            thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.25},daemon=True)
            thread.start();background.append((server,thread))
        urls=[f'http://{host}:{server.server_port}' for host,server in zip(hosts,servers)]
        print('VolleyMole 网页工作台：\n'+'\n'.join(urls)+f'\n工作区：{root}',flush=True)
        if args.open:
            import webbrowser
            webbrowser.open(urls[0])
        try: servers[0].serve_forever(poll_interval=.25)
        except KeyboardInterrupt: pass
    finally:
        for server,thread in background:
            server.shutdown();thread.join(timeout=2)
        for server in servers: server.server_close()
        app.jobs.close()


if __name__=='__main__': main()
