import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from volleymole.common import digest, read_json
from volleymole.meme_director import compile_plan, eligible_cue, policy, validate_decision
from volleymole.meme_stage import focused_indices, prepare, run_review
from volleymole import gemini_transport
from volleymole.llm_transport import ResponseContractError, TransportError
from volleymole.meme_verifier import observation_payload, validate_observation


def frames():
    return [{'id':f'f{i}', 'start_sec':i/10} for i in range(41)]


def clip(rid='a', offset=0):
    return {'id':rid, 'source_start_sec':0., 'source_end_sec':4., 'peak_sec':1.,
            'output_start_sec':offset, 'output_end_sec':offset+4., 'playback_rate':1.}


def assets():
    return {name: {'path':'/local/'+name+'.wav', 'ready':True, 'kind':'audio',
                   'source_start_sec':0., 'source_end_sec':length}
            for name,length in [('kaipao',1.65), ('nice',.84)]}


def decision(name='kaipao'):
    tags = {'full_arm_swing':['f5','f8'], 'forceful_contact':['f10'],
            'fast_post_contact_flight':['f11','f12','f13']} if name=='kaipao' else {
            'successful_outcome':['f12'], 'clean_execution':['f10'], 'controlled_save':['f10','f12']}
    return {'decision':'use', 'meme_id':name, 'confidence':.98,
            'action':'power_spike' if name=='kaipao' else 'dig',
            'power':'hard' if name=='kaipao' else 'not_applicable',
            'facts':[{'tag':tag,'frame_ids':refs,'description':'visible action evidence'} for tag,refs in tags.items()],
            'anchor_frame_id':'f14', 'placement':'after_action', 'contact_frame_ids':['f10'], 'reason':'complete visual evidence'}


def skip():
    return {**decision(), 'decision':'skip','meme_id':None,'anchor_frame_id':None,'placement':None,
            'facts':[], 'contact_frame_ids':[], 'reason':'insufficient evidence'}


def observation(action='power_spike', force='visibly_hard'):
    return {'target_player':'player next to ball', 'observed_motion':'visible movement',
            'ball_trajectory_before':'incoming trajectory', 'ball_trajectory_after':'outgoing trajectory',
            'visible_contact_description':'contact posture', 'action':action, 'force':force}


class MemePolicyTests(unittest.TestCase):
    def test_focus_sampling_keeps_all_contact_frames_and_both_context_edges(self):
        evidence=[{'id':f'f{i}','start_sec':100+i/12} for i in range(97)]
        selected=focused_indices(evidence,103.975)
        self.assertEqual(selected,sorted(set(selected)))
        self.assertIn(0,selected);self.assertIn(96,selected)
        self.assertLess(len(selected),len(evidence)//2)
        for i,frame in enumerate(evidence):
            if abs(frame['start_sec']-103.975)<=.6:self.assertIn(i,selected)

    def test_heavy_spike_requires_all_three_phases_not_just_attack_or_jump(self):
        self.assertEqual(eligible_cue(decision(),frames(),clip(),assets(),policy())[1], 'eligible')
        for action,power in [('tip','soft'),('roll_shot','soft'),('set','hard'),('power_spike','unknown')]:
            raw=decision();raw.update(action=action,power=power)
            self.assertIsNone(eligible_cue(raw,frames(),clip(),assets(),policy())[0])
        raw=decision();raw['facts']=raw['facts'][:2]
        self.assertEqual(eligible_cue(raw,frames(),clip(),assets(),policy())[1], 'missing_required_evidence')
        raw=decision();raw['facts'][2]['frame_ids']=['f11']
        self.assertEqual(eligible_cue(raw,frames(),clip(),assets(),policy())[1], 'missing_ordered_power_evidence')

    def test_user_tip_correction_vetoes_even_confident_wrong_model(self):
        corrected={**clip(), 'editor_constraints':{'action':'tip','forbidden_memes':['kaipao']}}
        cue,reason=eligible_cue(decision(),frames(),corrected,assets(),policy())
        self.assertIsNone(cue);self.assertEqual(reason,'editor_veto')

    def test_praise_waits_for_success_and_never_covers_contact_or_truncates_phrase(self):
        raw=decision('nice');raw['facts'][0]['frame_ids']=['f18']
        self.assertEqual(eligible_cue(raw,frames(),clip(),assets(),policy())[1],'outcome_not_yet_visible')
        raw=decision('nice');raw['anchor_frame_id']='f10'
        self.assertEqual(eligible_cue(raw,frames(),clip(),assets(),policy())[1],'would_cover_ball_contact')
        raw=decision('nice');raw['anchor_frame_id']='f35'
        self.assertEqual(eligible_cue(raw,frames(),clip(),assets(),policy())[1],'phrase_would_be_truncated')
        raw=decision('nice');raw['facts'].append({'tag':'failed_outcome','frame_ids':['f12'],'description':'failed'})
        self.assertEqual(eligible_cue(raw,frames(),clip(),assets(),policy())[1],'forbidden_condition')

    def test_unknown_frames_and_nonsense_confidence_are_rejected(self):
        for changes in [{'anchor_frame_id':'invented'}, {'confidence':float('nan')}, {'contact_frame_ids':[]}]:
            raw={**decision(),**changes}
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_decision(raw,frames(),policy())

    def test_four_rallies_no_repeated_praise_and_no_forced_replacement(self):
        jobs=[{'clip':clip(str(i),i*10),'evidence':frames()} for i in range(4)]
        results={str(i):decision('nice') for i in range(4)}
        plan=compile_plan(jobs,results,assets(),policy())
        self.assertEqual(len(plan['cues']),1)
        self.assertEqual(len(compile_plan(jobs,{str(i):skip() for i in range(4)},assets(),policy())['cues']),0)
        self.assertEqual(eligible_cue(decision(),frames(),clip(),{},policy())[1],'asset_unavailable')
        results['1']=decision();results['2']=decision()
        self.assertEqual(len(compile_plan(jobs,results,assets(),policy(),
            observations={'1':observation(),'2':observation()})['cues']),2)

    def test_kaipao_needs_independent_action_review_without_meme_priming(self):
        jobs=[{'clip':clip(),'evidence':frames()}]
        results={'a':decision()}
        missing=compile_plan(jobs,results,assets(),policy())
        self.assertFalse(missing['review_complete']);self.assertEqual(missing['cues'],[])
        disputed=compile_plan(jobs,results,assets(),policy(),observations={'a':observation('dig','not_determinable')})
        self.assertTrue(disputed['review_complete']);self.assertEqual(disputed['cues'],[])
        self.assertEqual(disputed['decisions'][0]['status'],'independent_action_disagrees')
        uncertain=compile_plan(jobs,results,assets(),policy(),observations={'a':observation('power_spike','not_determinable')})
        self.assertEqual(uncertain['cues'],[])

    def test_all_thirteen_have_positive_and_negative_conditions_and_story_is_not_invented(self):
        rules=policy()['rules'];self.assertEqual(len(rules),13)
        self.assertTrue(all(r['require_all'] and r['condition'] and r['avoid'] for r in rules))
        rule=next(r for r in rules if r['id']=='zhenxiang')
        raw=decision('nice');raw['meme_id']='zhenxiang'
        raw['facts']=[{'tag':t,'frame_ids':['f12'],'description':'visible'} for t in rule['require_all']]
        self.assertEqual(eligible_cue(raw,frames(),clip(),assets(),policy())[1],'missing_story_context')


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class MemeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.video=self.root/'source.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=160x240:rate=30:duration=4',
            '-f','lavfi','-i','sine=frequency=400:sample_rate=48000:duration=4',
            '-c:v','libx264','-preset','ultrafast','-c:a','aac',str(self.video)],check=True)
        (self.root/'assets.json').write_text(json.dumps({'assets':[{'id':'nice','path':'source.mp4','ready':True,
            'kind':'audio','source_start_sec':0.,'source_end_sec':.3}]}))
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'version':1,'video':'source.mp4','assets':'assets.json',
            'clips':[{**clip(),'source':'source.mp4','context_start_sec':0.,'context_end_sec':4.}]}))
        self.directory=self.root/'prepared'
        self.prepared=prepare(self.manifest,self.directory,width=960)

    def tearDown(self):self.temp.cleanup()

    def test_prepare_is_offline_run_caches_and_zero_cues_copy_original(self):
        from volleymole.meme_audio import render
        self.assertEqual(self.prepared['model_calls'],0)
        request=Mock(return_value=(skip(),{}))
        first=run_review(self.directory,'test-secret',1,request_fn=request)
        self.assertTrue(first['review_complete']);self.assertEqual(first['cues'],[])
        second=run_review(self.directory,'test-secret',1,request_fn=request)
        self.assertEqual(first,second);self.assertEqual(request.call_count,1)
        request_body=request.call_args.args[2]
        self.assertEqual(request_body['max_tokens'],8192)
        self.assertEqual(request_body['model'],'qwen3.5-35b-a3b')
        self.assertEqual(self.prepared['settings']['protocol'],'chat')
        contract=request_body['messages'][1]['content'][1]['text']
        self.assertEqual(json.loads(contract.removeprefix('Required JSON schema: ')),
                         request_body['response_format']['json_schema']['schema'])
        self.assertEqual(request_body['reasoning_effort'],'high')
        context=json.loads(request_body['messages'][1]['content'][0]['text'])
        self.assertEqual(context['focus_event_source_sec'],1.)
        self.assertIn('SAME FRAME DETAIL',context['image_layout'])
        self.assertEqual([r['id'] for r in context['policy']['rules']],['nice'])
        self.assertNotIn(str(self.root),json.dumps(request_body))
        self.assertNotIn('test-secret',(self.directory/'audio_plan.json').read_text())
        output=self.root/'unchanged.mp4';render(self.video,self.directory/'audio_plan.json',output)
        self.assertEqual(digest(self.video),digest(output))
        with self.assertRaisesRegex(ValueError,'budget'):
            run_review(self.directory,'test-secret',2,request_fn=request)

    def test_failure_halts_persistently_without_retry_or_fallback(self):
        request=Mock(side_effect=TransportError('timeout',retryable=True))
        plan=run_review(self.directory,'test-secret',1,request_fn=request)
        self.assertFalse(plan['review_complete']);self.assertEqual(plan['cues'],[])
        self.assertEqual(request.call_count,1)
        with self.assertRaisesRegex(ValueError,'halted'):
            run_review(self.directory,'test-secret',1,request_fn=request)
        self.assertEqual(request.call_count,1)

    def test_actual_api_anchor_maps_to_plan_without_manual_time(self):
        evidence=self.prepared['jobs'][0]['evidence']
        near=lambda t:min(evidence,key=lambda f:abs(f['start_sec']-t))['id']
        raw=decision('nice');raw['anchor_frame_id']=near(1.5);raw['contact_frame_ids']=[near(1.)]
        for fact in raw['facts']:fact['frame_ids']=[near(1.25)]
        plan=run_review(self.directory,'test-secret',1,request_fn=Mock(return_value=(raw,{})))
        self.assertEqual(len(plan['cues']),1)
        chosen=next(f['start_sec'] for f in evidence if f['id']==raw['anchor_frame_id'])
        self.assertEqual(plan['cues'][0]['at_sec'],chosen)

    def test_invalid_answer_keeps_redacted_diagnostic_and_never_becomes_a_cue(self):
        raw=skip();raw['facts']=[{'tag':'invented_condition',
            'frame_ids':[self.prepared['jobs'][0]['evidence'][0]['id']], 'description':'unsupported observation'}]
        plan=run_review(self.directory,'test-secret',1,request_fn=Mock(return_value=(raw,{})))
        self.assertFalse(plan['review_complete']);self.assertEqual(plan['cues'],[])
        saved=read_json(self.directory/'results/a.json')
        self.assertFalse(saved['reusable']);self.assertEqual(saved['raw'],raw)
        self.assertEqual(saved['failure']['validation_error'],'meme_response_fact')
        self.assertNotIn('test-secret',json.dumps(saved))

    def test_truncated_api_output_keeps_diagnostics_and_cannot_generate_a_cue(self):
        error=ResponseContractError('incomplete_response','length')
        error.diagnostics={'phase':'read','http_status':200,'finish_reason':'length',
                           'usage':{'completion_tokens':16384}}
        error.final_text='{"decision":"use","meme_id":'
        request=Mock(side_effect=error)
        plan=run_review(self.directory,'test-secret',1,request_fn=request)
        saved=read_json(self.directory/'results/a.json')
        self.assertFalse(plan['review_complete']);self.assertEqual(plan['cues'],[])
        self.assertEqual(saved['raw'],error.final_text)
        self.assertEqual(saved['failure']['transport']['usage']['completion_tokens'],16384)
        with self.assertRaisesRegex(ValueError,'halted'):
            run_review(self.directory,'test-secret',1,request_fn=request)
        self.assertEqual(request.call_count,1)

    def heavy_request(self):
        (self.root/'assets.json').write_text(json.dumps({'assets':[{'id':'kaipao','path':'source.mp4','ready':True,
            'kind':'audio','source_start_sec':0.,'source_end_sec':1.65}]}))
        directory=self.root/'heavy_prepared'
        prepared=prepare(self.manifest,directory,width=960)
        evidence=prepared['jobs'][0]['evidence']
        near=lambda t:min(evidence,key=lambda f:abs(f['start_sec']-t))['id']
        raw=decision();raw['anchor_frame_id']=near(1.5);raw['contact_frame_ids']=[near(1.)]
        for fact in raw['facts']:fact['frame_ids']=[near(int(ref[1:])/10) for ref in fact['frame_ids']]
        return directory,prepared,raw

    def test_independent_observation_uses_shared_budget_and_caches_both_stages(self):
        directory,prepared,raw=self.heavy_request()
        request=Mock(side_effect=[(raw,{}),(observation('dig','not_determinable'),{})])
        plan=run_review(directory,'test-secret',2,request_fn=request)
        self.assertTrue(plan['review_complete']);self.assertEqual(plan['cues'],[])
        self.assertEqual(plan['decisions'][0]['status'],'independent_action_disagrees')
        self.assertEqual(plan['api_calls_consumed'],2)
        wire=request.call_args.args[2]
        self.assertEqual(wire,observation_payload(prepared,prepared['jobs'][0],directory))
        for forbidden in ('kaipao','meme_id','available_memes','policy','test-secret'):
            self.assertNotIn(forbidden,json.dumps(wire))
        self.assertEqual(len([p for p in wire['messages'][1]['content'] if p['type']=='image_url']),7)
        self.assertEqual(run_review(directory,'test-secret',2,request_fn=request),plan)
        self.assertEqual(request.call_count,2)

    def test_missing_budget_or_failed_verifier_never_releases_kaipao(self):
        directory,_,raw=self.heavy_request()
        request=Mock(return_value=(raw,{}))
        plan=run_review(directory,'test-secret',1,request_fn=request)
        self.assertFalse(plan['review_complete']);self.assertEqual(plan['cues'],[])
        self.assertEqual(request.call_count,1)
        self.assertEqual(run_review(directory,'test-secret',1,request_fn=request),plan)
        self.assertEqual(request.call_count,1)

    def test_verifier_transport_failure_halts_without_retry(self):
        directory,_,raw=self.heavy_request()
        request=Mock(side_effect=[(raw,{}),TransportError('timeout',retryable=True)])
        plan=run_review(directory,'test-secret',2,request_fn=request)
        self.assertFalse(plan['review_complete']);self.assertEqual(plan['cues'],[])
        saved=read_json(directory/'verifications/a.json')
        self.assertEqual(saved['status'],'failed')
        self.assertEqual(saved['reservation']['stage'],'action_verification')
        with self.assertRaisesRegex(ValueError,'halted'):run_review(directory,'test-secret',2,request_fn=request)
        self.assertEqual(request.call_count,2)


class GeminiMemeTransportTests(unittest.TestCase):
    def payload(self):
        from volleymole.meme_director import response_schema
        return {'model':'gemini-3.1-pro-preview','max_tokens':8192,
                'messages':[{'role':'system','content':'test'},{'role':'user','content':[{'type':'text','text':'test'}]}],
                'response_format':{'json_schema':{'schema':response_schema()}}}

    def test_native_schema_and_only_final_answer_no_thought_retention(self):
        response=io.BytesIO(('data: '+json.dumps({'candidates':[{'finishReason':'STOP','content':{'parts':[
            {'thought':True,'text':'private thought'}, {'text':json.dumps(skip())}]}}],
            'usageMetadata':{'promptTokenCount':20,'private':'never-save'}})+'\n\n').encode())
        response.headers={'Content-Type':'text/event-stream'}
        opener=Mock(return_value=response)
        answer,metadata=gemini_transport.request_json('https://aihubmix.com/v1','test-secret',self.payload(),2,opener=opener)
        self.assertEqual(answer,skip());self.assertNotIn('private',json.dumps(metadata))
        request=opener.call_args.args[0]
        self.assertIn('/gemini/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse',request.full_url)
        self.assertIn('responseJsonSchema',json.loads(request.data)['generationConfig'])
        self.assertTrue(json.loads(request.data)['generationConfig']['thinkingConfig']['includeThoughts'])
        self.assertNotIn('test-secret',request.full_url)

    def test_truncated_response_and_redirect_are_not_success(self):
        response=io.BytesIO(('data: '+json.dumps({'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[{'text':'{}'}]}}]})+'\n\n').encode())
        response.headers={'Content-Type':'text/event-stream'}
        with self.assertRaises(ResponseContractError):
            gemini_transport.request_json('https://aihubmix.com/v1','test-secret',self.payload(),2,opener=Mock(return_value=response))
        self.assertIsNone(gemini_transport.NoRedirect().redirect_request(None,None,302,'',{},'https://elsewhere.invalid'))

    def test_native_stream_handles_chunks_heartbeats_and_requires_clean_finish(self):
        chunks=[{'candidates':[{'content':{'parts':[{'text':'{"ok":'}]}}]},
                {'candidates':[{'content':{'parts':[{'text':'true}'}]},'finishReason':'STOP'}]}]
        wire=': heartbeat\n\n'+''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)
        self.assertEqual(gemini_transport.decode_stream(wire.encode())[0],{'ok':True})
        for broken in [wire[:-1], ': heartbeat\n\n', wire+'data: '+json.dumps(chunks[0])+'\n\n']:
            with self.assertRaises(ResponseContractError):gemini_transport.decode_stream(broken.encode())


if __name__=='__main__':unittest.main()
