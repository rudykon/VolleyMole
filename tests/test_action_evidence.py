import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from volleymole.action_evidence import validate_action_evidence,load_action_evidence,context_hypotheses,uncovered_action_contexts


class ActionEvidenceTests(unittest.TestCase):
    def source(self):return {'identity':{'sha256':'a'*64},'duration_sec':20.}
    def data(self):
        return {'schema_version':1,'source':{'sha256':'a'*64,'duration_sec':20.,'time_basis':'source_relative_to_container_start'},
            'model':{'sha256':'b'*64,'manifest_sha256':'c'*64},'status':'complete',
            'covered_start_sec':0.,'covered_end_sec':20.,
            'events':[{'label':'spike','time_sec':5.,'raw_probability':.8,'observation_status':'candidate'}]}

    def test_source_and_model_evidence_is_checked_before_use(self):
        self.assertEqual(validate_action_evidence(self.data(),self.source())['status'],'complete')
        for path,value in [(('source','sha256'),'d'*64),(('source','duration_sec'),10),
                           (('source','time_basis'),'absolute_pts'),(('model','sha256'),'bad')]:
            data=self.data();data[path[0]][path[1]]=value
            with self.assertRaises(ValueError):validate_action_evidence(data,self.source())

    def test_unknown_or_partial_output_cannot_invent_predictions(self):
        data=self.data();data['status']='unsupported'
        with self.assertRaises(ValueError):validate_action_evidence(data,self.source())
        data['events']=[];validate_action_evidence(data,self.source())
        data=self.data();data.update(status='partial',covered_start_sec=0,covered_end_sec=4)
        with self.assertRaises(ValueError):validate_action_evidence(data,self.source())
        data['covered_end_sec']=6;validate_action_evidence(data,self.source())
        data.update(status='complete',covered_start_sec=None)
        with self.assertRaises(ValueError):validate_action_evidence(data,self.source())
        data.update(status='partial',events=[],covered_start_sec=None,covered_end_sec=None)
        self.assertEqual(validate_action_evidence(data,self.source())['events'],[])

    def test_label_is_a_hint_without_new_evidence_id(self):
        rows=context_hypotheses(self.data(),4,6)
        self.assertEqual(rows[0]['status'],'unverified_visual_model_hypothesis')
        self.assertNotIn('id',rows[0]);self.assertNotIn('evidence_ids',rows[0])
        self.assertEqual(context_hypotheses(self.data(),6,9),[])

    def test_existing_reviews_cover_hints_without_duplicate_work(self):
        self.assertEqual(uncovered_action_contexts(self.data(),[{'start':4,'end':6}],20),[])
        jobs=uncovered_action_contexts(self.data(),[],20)
        self.assertEqual([(j['start'],j['end']) for j in jobs],[(3.,7.)])

    def test_per_source_directory_never_reuses_another_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/('a'*64+'.json')).write_text(json.dumps(self.data()))
            args=SimpleNamespace(action_evidence_dir=root,action_evidence=None)
            data,signature=load_action_evidence(args,self.source())
            self.assertEqual(data['status'],'complete');self.assertEqual(len(signature),64)
            other={**self.source(),'identity':{'sha256':'d'*64}}
            self.assertEqual(load_action_evidence(args,other)[0]['status'],'not_found')


if __name__=='__main__':unittest.main()
