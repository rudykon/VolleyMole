"""Incremental bounded SSE primitives extracted from the validated replay transport."""
import json
import math
import time
from urllib.request import HTTPRedirectHandler
from .llm_transport import _strict_json
ENDPOINT='https://aihubmix.com/v1'
MODEL='gemini-3.1-pro-preview'
MAX_WIRE_BYTES=16*1024*1024
MAX_EVENT_BYTES=1024*1024
MAX_CONTENT_BYTES=1024*1024
MAX_EVENTS=100000
ALLOWED_CAUSES={'RemoteDisconnected','IncompleteRead','ConnectionResetError','ConnectionAbortedError','ConnectionRefusedError',
 'BrokenPipeError','TimeoutError','SSLError','SSLEOFError','SSLCertVerificationError','URLError','HTTPError','gaierror',
 'OSError','ConnectionError','HTTPException'}
REASONS={'timeout','connection_error','http_error','io_error','invalid_content_type','invalid_stream_event',
 'provider_stream_error','response_too_large','event_too_large','too_many_events','invalid_utf8',
 'incomplete_response','unsupported_choice','content_after_finish','refused','invalid_json','invalid_stream_field'}
class SSEError(RuntimeError):
    def __init__(self,reason,diagnostics):
        self.reason=reason if reason in REASONS else 'invalid_stream_event'
        self.diagnostics=diagnostics
        super().__init__(self.reason)

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None

def _audit(state,clock):
    out={k:state[k] for k in ('received_bytes','event_count','heartbeat_count','content_bytes','reasoning_delta_count','phase','request_bytes','image_count','http_status','model_version','response_id','clean_eof') if k in state}
    out['elapsed_sec']=round(max(0.,clock()-state['began']),6)
    for key in ('headers_received_sec','first_byte_sec','first_event_sec','first_content_sec','last_byte_sec'):
        out[key]=state.get(key)
    out['reasoning_content_retained']=False
    return out

def _cause(error):
    item=dict(cause_class=type(error).__name__ if type(error).__name__ in ALLOWED_CAUSES else 'other')
    number=getattr(error,'errno',None)
    if type(number) is int:item['errno']=number
    inner=getattr(error,'reason',None)
    if isinstance(inner,BaseException):
        item['underlying_cause_class']=type(inner).__name__ if type(inner).__name__ in ALLOWED_CAUSES else 'other'
        number=getattr(inner,'errno',None)
        if type(number) is int:item['underlying_errno']=number
    return item

def _fail(reason,state,clock,**safe):
    raise SSEError(reason,{**_audit(state,clock),**safe})

def _remaining(state,deadline,clock):
    left=deadline-clock()
    if left<=0:_fail('timeout',state,clock)
    return left

def _pop_line(buffer,final=False):
    positions=[p for p in (buffer.find(b'\n'),buffer.find(b'\r')) if p>=0]
    if not positions:return None
    end=min(positions)
    if buffer[end:end+1]==b'\r':
        if end+1==len(buffer) and not final:return None
        size=2 if buffer[end+1:end+2]==b'\n' else 1
    else:size=1
    return buffer[:end],buffer[end+size:]

def _events(response,state,deadline,clock,limits):
    buffer=b'';data=[];event=b'';event_size=0;first_line=True
    reader=getattr(response,'read1',None) or response.read
    while True:
        left=_remaining(state,deadline,clock)
        sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
        if sock is not None:sock.settimeout(left)
        chunk=reader(min(16384,limits['wire']+1-state['received_bytes']))
        _remaining(state,deadline,clock)
        if not isinstance(chunk,bytes):_fail('invalid_stream_field',state,clock)
        final=not chunk
        if chunk:
            now=round(clock()-state['began'],6)
            state.setdefault('first_byte_sec',now);state['last_byte_sec']=now
            state['received_bytes']+=len(chunk)
            if state['received_bytes']>limits['wire']:_fail('response_too_large',state,clock)
            buffer+=chunk
            if state.get('_progress'):state['_progress'](_audit(state,clock))
        while True:
            popped=_pop_line(buffer,final)
            if popped is None:break
            line,buffer=popped
            if first_line:
                if line.startswith(b'\xef\xbb\xbf'):line=line[3:]
                first_line=False
            if not line:
                if data:
                    state['event_count']+=1
                    if state['event_count']>limits['events']:_fail('too_many_events',state,clock)
                    state.setdefault('first_event_sec',round(clock()-state['began'],6))
                    yield event,b'\n'.join(data)
                data=[];event=b'';event_size=0
            elif line.startswith(b':'):
                state['heartbeat_count']+=1
            else:
                field,sep,value=line.partition(b':')
                if value.startswith(b' '):value=value[1:]
                if field==b'data':
                    data.append(value);event_size+=len(value)+1
                elif field==b'event':event=value
                elif field in (b'id',b'retry'):pass  # Never persist trace IDs or provider retry advice.
            if event_size>limits['event']:_fail('event_too_large',state,clock)
        if event_size+len(buffer)>limits['event']:_fail('event_too_large',state,clock)
        if final:
            # Reject unterminated event data even after a previous STOP.
            if data or buffer.strip():_fail('incomplete_response',state,clock)
            state['clean_eof']=True
            return

def safe_http_body(error,key):
    """Keep structural error fields only, never free text, credentials or echoes."""
    import re
    try:
        data=_strict_json(error.read(16384),'invalid_json')
        item=data.get('error',{}) if isinstance(data,dict) else {}
        out={'body_available':True,'json_error':isinstance(item,dict)}
        if isinstance(item,dict):
            for name in ('type','code','param'):
                value=item.get(name)
                if type(value) is int:out[name]=value
                elif isinstance(value,str) and not (key and key in value) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}',value):out[name]=value
        return out
    except Exception:return {'body_available':False}
    finally:error.close()
