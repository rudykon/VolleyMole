import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from volleymole.common import APP,identity,probe,read_json,save_json
from volleymole.event_pipeline import Discovery
from volleymole.run_match import argument_parser


class ImportedActionPipelineTests(unittest.TestCase):
    def test_same_source_hypothesis_reaches_visual_review_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'fixture.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=25:duration=4',
                '-c:v','libx264','-threads','1',str(video)],check=True)
            source=probe(video);source['identity']=identity(video)
            proposal={'schema_version':1,'source':{'sha256':source['identity']['sha256'],'duration_sec':4.,
                'time_basis':'source_relative_to_container_start'},
                'model':{'sha256':'b'*64,'manifest_sha256':'c'*64},'status':'complete',
                'covered_start_sec':0.,'covered_end_sec':4.,
                'events':[{'label':'spike','time_sec':1.,'raw_probability':.8,'observation_status':'candidate'}]}
            path=root/'actions.json';save_json(path,proposal)
            args=argument_parser().parse_args(['--video',str(video),'--model','fake',
                '--action-evidence',str(path),'--analysis-cache-dir',str(root/'cache'),'--no-sound-model'])
            manifest={'rallies':[],'config':read_json(APP/'defaults.json')}
            calls=[]
            def respond(endpoint,key,payload,timeout):
                evidence=json.loads(payload['messages'][1]['content'][0]['text'])
                calls.append(evidence)
                self.assertEqual(evidence['local']['action_hypotheses'][0]['label'],'spike')
                self.assertFalse(any(e['kind']=='local_action_model' for e in evidence['evidence']))
                return {'events':[]},{'model':'fake','finish_reason':'stop'}
            with patch.dict(os.environ,{'VOLLEYMOLE_API_KEY':'test-no-network'}), \
                 patch('volleymole.semantic.request_json',side_effect=respond):
                report=Discovery(source,root,args).finish(manifest)
                self.assertEqual(report['local_action_candidates'],1)
                self.assertEqual(report['events'],[])
                self.assertEqual([c['phase'] for c in calls],['coarse','review'])
                Discovery(source,root,args).finish(manifest)
                self.assertEqual(len(calls),2)
                proposal['events'][0]['raw_probability']=.7;save_json(path,proposal)
                Discovery(source,root,args).finish(manifest)
                self.assertEqual(len(calls),4)
                args.max_review_sec=3.
                long_rally={'rally_id':'too_long','start_sec':0.,'end_sec':4.,'safe_start_sec':0.,'safe_end_sec':4.}
                report=Discovery(source,root,args).finish({**manifest,'rallies':[long_rally]})
                self.assertEqual(report['review_completed'],1)
                self.assertTrue(any(f['job']=='too_long' for f in report['failures']))
                self.assertEqual(report['local_action_coverage']['covered_end_sec'],4.)


if __name__=='__main__':unittest.main()
