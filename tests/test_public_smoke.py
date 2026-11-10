import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fineval_pipeline.agent import parse_findings,static_candidates
from scripts.public_release_check import check

ROOT=Path(__file__).resolve().parents[1]

class PublicSourceTests(unittest.TestCase):
    def test_release_contains_source_only(self):
        self.assertFalse(check(ROOT)['weights_in_git'])

    def test_static_candidate_has_original_line_and_evidence(self):
        source='from Crypto.Cipher import AES\nc = AES.new(b"k"*16, AES.MODE_ECB)\n'
        candidates=static_candidates(source,'example.py')
        self.assertTrue(any(x['rule_id']=='AES_ECB' and x['line']==2 for x in candidates))

    def test_dpo_plan_is_local_and_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';data=Path(tmp)/'train.jsonl'
            data.write_text(json.dumps({'messages':[{'role':'user','content':'Example'}],'rejected_response':'Example'})+'\n')
            run=subprocess.run([sys.executable,str(ROOT/'finsec.py'),'train-plan','--stage','dpo',
                '--model-dir',str(Path(tmp)/'base'),'--parent',str(Path(tmp)/'parent'),'--data',str(data),'--output',str(output)],
                cwd=ROOT,text=True,capture_output=True,check=True)
            plan=json.loads((output/'train_config.json').read_text())
            self.assertEqual(plan['arguments']['beta'],.1)
            self.assertEqual(plan['arguments']['rpo_alpha'],.2)
            self.assertFalse(json.loads(run.stdout)['execution_requested'])

    def test_invalid_source_evidence_is_rejected(self):
        finding={'rule_id':'AES_ECB','file':'a.py','line':99,'evidence':'fake','reason':'example','fix':'example'}
        with self.assertRaises(ValueError):parse_findings(json.dumps({'findings':[finding]}),'a.py','pass\n',{'AES_ECB'})

if __name__=='__main__':unittest.main()
