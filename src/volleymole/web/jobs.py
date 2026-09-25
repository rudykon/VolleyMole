"""Persistent serial queue, isolated process groups and bounded log reads."""
import copy
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

ACTIVE = {'queued', 'running', 'cancelling'}


def process_stamp(pid):
    """Linux start ticks distinguish an owned job from a subsequently reused PID."""
    try:
        fields=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (ValueError,OSError,IndexError,TypeError):
        return None


class RecoveredProcess:
    def __init__(self,pid,stamp):
        self.pid=pid
        self.stamp=stamp

    def wait(self):
        while process_stamp(self.pid)==self.stamp:
            time.sleep(.25)
        # A non-child's exit status cannot be recovered. Do not invent success.
        return None


def redact(value):
    if isinstance(value,dict):
        return {k:('[已隐藏]' if re.search(r'api.?key|token|secret|password|authorization',k,re.I) and k not in {'max_tokens','semantic_max_tokens','max_output_tokens'} else redact(v)) for k,v in value.items()}
    if isinstance(value,list): return [redact(v) for v in value]
    if isinstance(value,str):
        value=re.sub(r'(?i)(bearer\s+)\S+',r'\1[已隐藏]',value)
        value=re.sub(r'\bsk-[A-Za-z0-9_-]{8,}', '[已隐藏]', value)
        value=re.sub(r'(?i)(["\']?(?:api_key|api-key|authorization|password)["\']?\s*[:=]\s*["\']?)[^"\'\s,}]+',r'\1[已隐藏]',value)
    return value


class JobManager:
    def __init__(self, workspace):
        self.workspace=Path(workspace)
        self.directory=self.workspace/'.local/web'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.file_lock=(self.directory/'server.lock').open('a')
        try: fcntl.flock(self.file_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.file_lock.close()
            raise RuntimeError('此工作区已有网页服务运行') from None
        self.lock=threading.RLock()
        self.jobs={}
        self.processes={}
        self.recovered={}
        self.cancellations={}
        self.queue=queue.Queue()
        self.stopping=False
        for p in self.directory.glob('*.json'):
            try:
                record=json.loads(p.read_text())
                if record['status'] in {'running','cancelling'}:
                    stamp=record.get('process_stamp')
                    if stamp and process_stamp(record.get('pid'))==stamp:
                        self.recovered[record['id']]=RecoveredProcess(record['pid'],stamp)
                        record.update(status='running',note='已重新连接仍在运行的进程')
                    else:
                        record.update(status='interrupted',finished_at=time.time(),error='原进程已结束，退出码不可恢复；请检查已有产物后重试')
                self.jobs[record['id']]=record
                self._save(record)
            except (ValueError,KeyError,OSError): continue
        pending=sorted(self.jobs.values(),key=lambda j:(j['id'] not in self.recovered,j['created_at']))
        for record in pending:
            if record['status'] in ACTIVE: self.queue.put(record['id'])
        self.worker=threading.Thread(target=self._worker,daemon=True,name='web-job-queue')
        self.worker.start()

    def _save(self,record):
        p=self.directory/(record['id']+'.json')
        tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False))
        tmp.replace(p)

    def list(self):
        with self.lock:
            return copy.deepcopy(sorted(self.jobs.values(),key=lambda r:r['created_at'],reverse=True))

    def get(self,job_id):
        with self.lock:
            if job_id not in self.jobs: raise ValueError('任务不存在')
            return copy.deepcopy(self.jobs[job_id])

    def submit(self,command,subcommand,argv,values,title=''):
        with self.lock:
            if self.stopping: raise ValueError('服务正在关闭')
            if sum(j['status'] in ACTIVE for j in self.jobs.values())>=50:
                raise ValueError('队列已满，请等待现有任务完成')
            job_id=uuid.uuid4().hex
            record={'id':job_id,'command':command,'subcommand':subcommand,'argv':argv,'values':values,
                    'title':str(title or command)[:120], 'status':'queued','created_at':time.time(),
                    'started_at':None,'finished_at':None,'returncode':None}
            self.jobs[job_id]=record
            self._save(record)
            self.queue.put(job_id)
            return copy.deepcopy(record)

    def _worker(self):
        while True:
            job_id=self.queue.get()
            if job_id is None: return
            try:
                with self.lock:
                    job=self.jobs[job_id]
                    if job['status']!='queued' and job_id not in self.recovered: continue
                    env=os.environ.copy()
                    env['PYTHONUNBUFFERED']='1'
                    env['VOLLEYMOLE_WORKSPACE']=str(self.workspace)
                    env['VOLLEYMOLE_TEMPLATES']=str(self.workspace/'templates')
                    if (self.workspace/'data/visual-assets').is_dir():
                        env.setdefault('VOLLEYMOLE_ASSETS',str(self.workspace/'data/visual-assets'))
                    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
                    process=self.recovered.pop(job_id,None)
                    if process is None:
                        with (self.directory/(job_id+'.log')).open('ab',buffering=0) as out:
                            process=subprocess.Popen([sys.executable,'-u','-m','volleymole',*job['argv']],
                                cwd=self.workspace,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,
                                start_new_session=True)
                    self.processes[job_id]=process
                    job.update(status='running',started_at=job.get('started_at') or time.time(),pid=process.pid,
                               process_stamp=process_stamp(process.pid))
                    self._save(job)
                code=process.wait()
                # Keep the queue blocked until all descendants receive SIGKILL,
                # even when the CLI parent exits immediately on SIGTERM.
                with self.lock:
                    cancellation=self.cancellations.get(job_id)
                if cancellation is not None: cancellation.wait(timeout=3)
                # Package finished films before releasing the queue. A failed
                # packaging step must not rewrite the CLI's real exit status.
                publication = {}
                if code == 0 and job['status'] != 'cancelling':
                    from .films import publish_job
                    try:
                        film_ids = publish_job(self.workspace, {**job, 'status': 'succeeded'}, self.log(job_id),
                                               cancelled=lambda: self.get(job_id)['status']=='cancelling')
                        publication = {'film_ids': film_ids}
                    except Exception as exc:
                        publication = {'film_ids': [], 'publication_error': '成片收录失败：' + redact(str(exc))}
                with self.lock:
                    job.update(status='cancelled' if job['status']=='cancelling' else 'interrupted' if code is None else 'succeeded' if code==0 else 'failed',
                               returncode=code,finished_at=time.time(), **publication)
                    if code is None and job['status']!='cancelled':
                        job['error']='原进程已结束，但服务中断期间的退出码不可恢复；请检查产物和验证报告'
                    self.processes.pop(job_id,None)
                    self.cancellations.pop(job_id,None)
                    self._save(job)
            except Exception as exc:
                with self.lock:
                    self.jobs[job_id].update(status='failed',error=str(exc),finished_at=time.time())
                    self._save(self.jobs[job_id])
            finally:
                self.queue.task_done()

    @staticmethod
    def _signal(process,sig):
        try: os.killpg(process.pid,sig)
        except ProcessLookupError: pass

    def cancel(self,job_id):
        with self.lock:
            self.get(job_id)
            job=self.jobs[job_id]
            if job['status']=='queued':
                job.update(status='cancelled',finished_at=time.time())
            elif job['status']=='running':
                job['status']='cancelling'
                process=self.processes.get(job_id) or self.recovered.get(job_id)
                if process is None: return copy.deepcopy(job)
                finished=threading.Event()
                self.cancellations[job_id]=finished
                self._signal(process,signal.SIGTERM)
                def finish_cancel():
                    # Also kill remaining descendants after the parent exits.
                    time.sleep(2)
                    try: self._signal(process,signal.SIGKILL)
                    finally: finished.set()
                threading.Thread(target=finish_cancel,daemon=True).start()
            self._save(job)
            return copy.deepcopy(job)

    def log(self,job_id):
        self.get(job_id)
        p=self.directory/(job_id+'.log')
        if not p.exists(): return ''
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-100_000))
            return redact(f.read(100_000).decode('utf-8',errors='replace'))

    def close(self):
        with self.lock:
            self.stopping=True
            for job_id in list(self.jobs):
                if self.jobs[job_id]['status'] in ACTIVE: self.cancel(job_id)
            self.queue.put(None)
        self.worker.join(timeout=5)
        self.file_lock.close()
