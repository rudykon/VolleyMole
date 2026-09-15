import copy
import importlib.util
from pathlib import Path
import unittest

from volleymole.blooper_benchmark import DIMENSIONS,FLAGS,STAGES,validate_forms

spec=importlib.util.spec_from_file_location('auto_bloopers',Path(__file__).resolve().parents[1]/'scripts/auto_annotate_bloopers.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class AutomaticBlooperTests(unittest.TestCase):
    def unknown(self):
        return {'observed_facts':[],'ratings':dict.fromkeys(DIMENSIONS),'flags':dict.fromkeys(FLAGS),
            'stages':{s:{'start_sec':None,'end_sec':None} for s in STAGES},'confidence':0,
            'uncertainty':'No reliable visible fact in this unit fixture.'}

    def test_unknown_is_valid_and_does_not_become_zero(self):
        value=self.unknown()
        result=mod.validate_prediction(value,[],0,4)
        self.assertTrue(all(v is None for v in result['ratings'].values()))
        value['ratings']['overall_fun']=0
        with self.assertRaises(ValueError):mod.validate_prediction(value,[],0,4)

    def test_facts_need_supplied_frame_ids_and_audio_stays_unknown(self):
        value=self.unknown();value['observed_facts']=[{'text':'Unit fixture fact','frame_ids':['frame_0']}]
        evidence=[{'id':'frame_0','kind':'frame','start_sec':1,'end_sec':1}]
        mod.validate_prediction(value,evidence,0,4)
        changed=copy.deepcopy(value);changed['observed_facts'][0]['frame_ids']=['invented_frame']
        with self.assertRaises(ValueError):mod.validate_prediction(changed,evidence,0,4)
        changed=copy.deepcopy(value);changed['ratings']['related_laughter']=0
        with self.assertRaises(ValueError):mod.validate_prediction(changed,evidence,0,4)

    def test_automatic_schema_cannot_claim_or_enter_human_ground_truth(self):
        record={'schema_version':mod.VERSION,'annotation_source':'model','is_ground_truth':False,'human_only':False,
            'annotation':self.unknown(),'evidence':[],'start_sec':0,'end_sec':4}
        mod.validate_record(record)
        for key,value in [('annotation_source','human'),('human_only',True),('is_ground_truth',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):mod.validate_record({**record,key:value})
        with self.assertRaises(ValueError):validate_forms({'assignments':[],'items':[]},[record])


if __name__=='__main__':unittest.main()
