"""Bounded OpenAI-compatible SSE JSON transport for long visual reviews.

Reasoning deltas keep the connection active but are never retained. A complete
stop event, DONE marker, and clean SSE end are mandatory before parsing JSON.
"""
import json,time
from http.client import HTTPException
from urllib.error import HTTPError,URLError
from urllib.request import Request,build_opener
from .gemini_stream import _events,NoRedirect,SSEError,safe_http_body
from .llm_transport import TransportError,ResponseContractError,parse_json_content,_strict_json,_safe_usage

def request_json(endpoint,key,payload,timeout,*,opener=None,clock=time.monotonic):
    if timeout<=0:raise ValueError('timeout must be positive')
    started=clock();deadline=started+timeout
    state=dict(began=started,received_bytes=0,event_count=0,heartbeat_count=0,content_bytes=0,reasoning_delta_count=0,phase='open')
    wire={**payload,'stream':True,'stream_options':{'include_usage':True}}
    req=Request(endpoint.rstrip('/')+'/chat/completions',data=json.dumps(wire,allow_nan=False).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'text/event-stream'})
    chunks=[];finish=None;done=False;usage=None;events=0
    try:
        with (opener or build_opener(NoRedirect()).open)(req,timeout=timeout) as response:
            if 'text/event-stream' not in response.headers.get('Content-Type','').lower():
                raise ResponseContractError('invalid_response_envelope')
            state['phase']='read'
            for _,data in _events(response,state,deadline,clock,{'wire':16*1024*1024,'event':1024*1024,'events':100000}):
                if data.strip()==b'[DONE]':
                    if done:raise ResponseContractError('invalid_response_envelope')
                    done=True;continue
                if done:raise ResponseContractError('invalid_response_envelope')
                obj=_strict_json(data,'invalid_response_envelope');events+=1
                if not isinstance(obj,dict) or obj.get('error'):raise ResponseContractError('invalid_response_envelope')
                if obj.get('usage') is not None:usage=_safe_usage(obj['usage'])
                choices=obj.get('choices',[])
                if not isinstance(choices,list) or len(choices)>1:raise ResponseContractError('invalid_response_envelope')
                if not choices:continue
                choice=choices[0]
                if choice.get('index',0)!=0:raise ResponseContractError('invalid_response_envelope')
                delta=choice.get('delta',{})
                if not isinstance(delta,dict):raise ResponseContractError('invalid_response_envelope')
                if delta.get('refusal'):raise ResponseContractError('refused')
                text=delta.get('content')
                if text is not None:
                    if not isinstance(text,str) or (finish is not None and text):raise ResponseContractError('invalid_response_envelope')
                    if text:
                        chunks.append(text);state['content_bytes']+=len(text.encode())
                        if state['content_bytes']>1024*1024:raise ResponseContractError('response_too_large')
                if delta.get('reasoning_content'):state['reasoning_delta_count']+=1
                reason=choice.get('finish_reason')
                if reason is not None:
                    if finish is not None or reason!='stop':raise ResponseContractError('incomplete_response',reason)
                    finish=reason
    except HTTPError as exc:
        status=exc.code;safe_http_body(exc,key);exc.close()
        raise TransportError('http_error',retryable=status in (408,429) or 500<=status<600,http_status=status) from None
    except TimeoutError:raise TransportError('timeout',retryable=True) from None
    except (URLError,ConnectionError,HTTPException):raise TransportError('connection_error',retryable=True) from None
    except SSEError as exc:
        if exc.reason=='timeout':raise TransportError('timeout',retryable=True) from None
        raise ResponseContractError('incomplete_response') from None
    if finish!='stop' or not done or not state.get('clean_eof'):raise ResponseContractError('incomplete_response',finish)
    raw,encoding=parse_json_content(''.join(chunks),finish)
    return raw,{'model':payload['model'],'finish_reason':finish,'usage':usage,'response_format':payload.get('response_format',{}).get('type'),
        'response_content_encoding':encoding,'streaming':True,'events':events,'reasoning_content_retained':False,
        'reasoning_delta_count':state['reasoning_delta_count'],'first_byte_sec':state.get('first_byte_sec'),'elapsed_sec':clock()-started}
