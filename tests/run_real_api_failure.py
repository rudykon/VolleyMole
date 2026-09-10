"""Real-video CLI regression against a local HTTP 503 endpoint (no live API outage claim)."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading

from volleymole.common import read_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--analysis-cache-dir',type=Path,required=True)
    parser.add_argument('--top-k',type=int,choices=[5,10],default=10)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    calls = []
    class Failure(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_POST(self):
            size = int(self.headers.get('Content-Length',0))
            if not 0 <= size <= 16*1024*1024:
                self.send_error(413)
                return
            left = size
            while left:
                block = self.rfile.read(min(left,65536))
                if not block:
                    break
                left -= len(block)
            calls.append({'path':self.path,'request_bytes':size,'received_bytes':size-left,'status':503})
            # Do not persist request bodies, Authorization headers or images.
            body = b'{"error":{"message":"controlled local test failure"}}'
            self.send_response(503)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1',0),Failure)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    env = dict(os.environ)
    env['VOLLEYMOLE_API_KEY'] = 'local-regression-key-not-a-real-credential'
    env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost'
    command = [sys.executable,'-m','volleymole','run','--video',str(args.video.resolve()),
        '--output',str(args.output),'--models',str(args.models.resolve()),
        '--analysis-cache-dir',str(args.analysis_cache_dir.resolve()),'--top-k',str(args.top_k),
        '--device',args.device,'--ranker','auto','--model','local-fault-model','--api-timeout','5',
        '--api-base',f'http://127.0.0.1:{server.server_port}/v1',
        '--llm-config',str(args.output/'intentionally-absent-llm-config.json'),'--rerun-from','rank']
    try:
        with (args.output/'fault-injection-cli.log').open('w') as log:
            result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    decision = read_json(args.output/'edit_decision.json') if (args.output/'edit_decision.json').is_file() else {}
    success = (result.returncode==0 and len(calls)>0 and decision.get('ranking_mode')=='rules_fallback'
               and decision.get('fallback',{}).get('http_status')==503)
    save_json(args.output/'api-failure-regression.json', {'status':'passed' if success else 'failed',
        'mechanism':'Real local HTTP 503 endpoint; no monkeypatch; not a real provider outage.',
        'video':str(args.video.resolve()),'calls':calls,'cli_returncode':result.returncode,
        'ranking_mode':decision.get('ranking_mode'),'fallback':decision.get('fallback'),
        'uploaded_to_external_provider':False})
    print('API failure real-video regression:', 'passed' if success else 'failed')
    if not success:
        raise SystemExit(1)


if __name__=='__main__':
    main()
