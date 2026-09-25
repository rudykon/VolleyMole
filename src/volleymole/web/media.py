"""Bounded, fingerprinted video metadata and real-frame thumbnail cache."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
from urllib.parse import quote

VIDEO={'.mp4','.mov','.mkv','.avi','.webm','.m4v','.mts','.m2ts'}
AUDIO={'.wav','.mp3','.m4a','.aac','.flac','.ogg'}


class MediaCache:
    def __init__(self,workspace):
        self.workspace=workspace
        self.directory=workspace.root/'.local/web/media'
        self.slots=threading.BoundedSemaphore(2)
        self.lock=threading.RLock()

    def source(self,value):
        path=self.workspace.media_path(value)
        if not path.is_file() or path.suffix.lower() not in VIDEO|AUDIO:
            raise ValueError('请选择视频或音频文件')
        stat=path.stat()
        key=hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}:v1'.encode()).hexdigest()
        return path,key

    def info(self,value):
        path,key=self.source(value)
        cache=self.directory/(key+'.json')
        # The limit also bounds concurrent ffprobe/ffmpeg processes. Recheck the
        # cache inside it so simultaneous views reuse an already completed frame.
        with self.slots:
            with self.lock:
                if cache.is_file():
                    try: return json.loads(cache.read_text())
                    except (ValueError,OSError): pass
            result={'duration':None,'width':None,'height':None,'poster':None,'status':'unavailable'}
            try:
                probe=subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],
                                     capture_output=True,text=True,timeout=15,check=True)
                data=json.loads(probe.stdout)
                streams=data.get('streams',[])
                video=next((s for s in streams if s.get('codec_type')=='video'),None)
                duration=float(data.get('format',{}).get('duration') or (video or {}).get('duration') or 0)
                result.update(status='ready',duration=duration if math.isfinite(duration) and duration>0 else None)
                if video:
                    width,height=int(video['width']),int(video['height'])
                    rotation=next((s.get('rotation',0) for s in video.get('side_data_list',[]) if 'rotation' in s),video.get('tags',{}).get('rotate',0))
                    if abs(round(float(rotation)))%180==90: width,height=height,width
                    result.update(width=width,height=height)
                    self.directory.mkdir(parents=True,exist_ok=True)
                    poster=self.directory/(key+'.jpg')
                    frame=subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-ss',str(min(duration*.15,8) if duration>0 else 0),
                                          '-i',str(path),'-frames:v','1','-vf',"scale='min(480,iw)':-2",'-threads','1',
                                          '-f','image2pipe','-vcodec','mjpeg','pipe:1'],capture_output=True,timeout=20)
                    if frame.returncode==0 and frame.stdout.startswith(b'\xff\xd8'):
                        with self.lock: poster.write_bytes(frame.stdout)
                        result['poster']='/api/thumbnail?path='+quote(self.workspace.relative(path),safe='')+'&v='+key
            except (OSError,ValueError,KeyError,subprocess.SubprocessError):
                result['note']='暂时无法读取媒体信息，可尝试下载原文件。'
            if result['status']=='ready':
                self.directory.mkdir(parents=True,exist_ok=True)
                with self.lock: cache.write_text(json.dumps(result))
            return result

    def thumbnail(self,value):
        _,key=self.source(value)
        path=self.directory/(key+'.jpg')
        if not path.is_file(): self.info(value)
        if not path.is_file(): raise FileNotFoundError('此文件暂无可用封面')
        return path
