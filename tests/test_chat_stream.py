import io,json,unittest
from volleymole.chat_stream import request_json
from volleymole.llm_transport import ResponseContractError

class Response(io.BytesIO):
    headers={'Content-Type':'text/event-stream'}

def stream(events,done=True):
    return b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)+(b'data: [DONE]\n\n' if done else b'')

class ChatStreamTests(unittest.TestCase):
    def call(self,data):
        return request_json('https://example.invalid/v1','not-a-key',{'model':'open-model'},10,opener=lambda *a,**k:Response(data))
    def test_complete_split_json_with_discarded_reasoning_and_usage(self):
        events=[{'choices':[{'delta':{'reasoning_content':'private'},'finish_reason':None}]},
          {'choices':[{'delta':{'content':'{"ok":'},'finish_reason':None}]},
          {'choices':[{'delta':{'content':'true}'},'finish_reason':'stop'}]},
          {'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':8}}]
        raw,meta=self.call(stream(events));self.assertEqual(raw,{'ok':True});self.assertEqual(meta['usage']['prompt_tokens'],10)
        self.assertEqual(meta['reasoning_delta_count'],1);self.assertNotIn('private',str(meta))
    def test_missing_done_is_not_accepted(self):
        with self.assertRaises(ResponseContractError):self.call(stream([{'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}]}],False))
    def test_length_finish_is_not_accepted(self):
        with self.assertRaises(ResponseContractError):self.call(stream([{'choices':[{'delta':{'content':'{}'},'finish_reason':'length'}]}]))
    def test_content_after_stop_is_not_accepted(self):
        with self.assertRaises(ResponseContractError):self.call(stream([{'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}]},{'choices':[{'delta':{'content':'x'}}]}]))
    def test_provider_error_is_not_parsed_as_model_json(self):
        with self.assertRaises(ResponseContractError):self.call(stream([{'error':{'message':'bad'}}]))
