import datetime
import io
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from zoneinfo import ZoneInfo

from fineval_pipeline.deepseek_api import BudgetLedger, BudgetExceeded, DeepSeek, APITransportError, rates, usage_cost


def reserve_worker(root):
    ledger=BudgetLedger(root,25.)
    for _ in range(20):
        try:ledger.reserve(.7,'test','attacker')
        except BudgetExceeded:break


class BudgetTests(unittest.TestCase):
    def test_explicit_unlimited_ledger_tracks_cost_and_keeps_scope(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=BudgetLedger(root,None)
            state=json.loads(ledger.path.read_text());state['allowed_variants']=['finance','hardened'];state['allowed_roles']=['attacker','search_judge']
            ledger.path.write_text(json.dumps(state))
            key=ledger.reserve(100.,'finance','attacker');ledger.settle(key,99.)
            self.assertEqual(ledger.snapshot()['spent_cny'],99.)
            self.assertIsNone(ledger.snapshot()['remaining_unreserved_cny'])
            with self.assertRaises(BudgetExceeded):ledger.reserve(1.,'teacher_generation','attacker')
            with self.assertRaises(BudgetExceeded):ledger.reserve(1.,'finance','training')
            with self.assertRaises(ValueError):BudgetLedger(root,20.)

    def test_process_shared_reservations_never_overspend(self):
        with tempfile.TemporaryDirectory() as root:
            BudgetLedger(root,25.)
            children=[multiprocessing.Process(target=reserve_worker,args=(root,)) for _ in range(4)]
            for child in children:child.start()
            for child in children:child.join();self.assertEqual(child.exitcode,0)
            state=BudgetLedger(root,25.).snapshot()
            self.assertLessEqual(state['committed_upper_cny'],25.)
            self.assertAlmostEqual(state['committed_upper_cny'],24.5)
            self.assertEqual(len(state['pending']),35)

    def test_settlement_refunds_only_known_cost_and_rejects_limit_change(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=BudgetLedger(root,25.)
            key=ledger.reserve(10.,'base_nf4','attacker');ledger.settle(key,1.)
            ledger.reserve(24.,'dpo','attacker')
            with self.assertRaises(BudgetExceeded):ledger.reserve(.001,'dpo','judge')
            with self.assertRaises(ValueError):BudgetLedger(root,26.)
            self.assertEqual(ledger.snapshot()['committed_upper_cny'],25.)

    def test_cache_tokens_and_peak_price(self):
        usage={'prompt_tokens':1000,'completion_tokens':500,'prompt_cache_hit_tokens':800,'prompt_cache_miss_tokens':200}
        self.assertAlmostEqual(usage_cost(usage,{'hit':.02,'miss':1.,'output':4.}),.002216)
        zone=ZoneInfo('Asia/Shanghai')
        self.assertEqual(rates(datetime.datetime(2026,11,8,10,tzinfo=zone))['output'],4.)
        self.assertEqual(rates(datetime.datetime(2026,11,9,10,tzinfo=zone))['output'],8.)
        with self.assertRaises(ValueError):usage_cost({**usage,'prompt_tokens':999},rates())

    def test_timeout_keeps_reservation_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'HB_API_BUDGET_ROOT':root,'HB_API_BUDGET_CNY':'25','DEEPSEEK_API_KEY':'fake-private-key'}):
            client=DeepSeek('base_nf4')
            with patch('urllib.request.urlopen',side_effect=urllib.error.URLError('timeout')) as request:
                with self.assertRaises(APITransportError):client.generate([{'role':'user','content':'hello'}],512,1.)
                self.assertEqual(request.call_count,1)
            state=client.ledger.snapshot();self.assertEqual(len(state['pending']),1)
            self.assertGreater(state['committed_upper_cny'],0)

    def test_user_disabled_api_cannot_make_requests_or_alter_existing_charges(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
            'HB_API_BUDGET_ROOT': root, 'HB_API_BUDGET_CNY': '20', 'DEEPSEEK_API_KEY': 'fake-private-key'}):
            ledger = BudgetLedger(root, 20.)
            known = ledger.reserve(1., 'already_completed', 'reviewer')
            ledger.settle(known, .25)
            ledger.reserve(.15, 'uncertain_past_request', 'attacker')
            state = json.loads(ledger.path.read_text())
            state['stop_reason'] = 'User disabled DeepSeek; evaluate GCG only'
            ledger.path.write_text(json.dumps(state))
            client = DeepSeek('must_not_start')
            before = client.ledger.snapshot()
            with patch('urllib.request.urlopen') as request:
                with self.assertRaises(BudgetExceeded):
                    client.generate([{'role': 'user', 'content': 'A benign test prompt.'}], 512, 0.)
                request.assert_not_called()
            self.assertEqual(client.ledger.snapshot(), before)
            self.assertEqual(before['spent_cny'], .25)
            self.assertEqual(len(before['pending']), 1)

    def test_transport_disables_thinking_and_logs_usage_without_key(self):
        body={'id':'request-test','model':'deepseek-flash','usage':{'prompt_tokens':100,'completion_tokens':10},
              'choices':[{'finish_reason':'stop','message':{'content':'{"improvement":"ok","prompt":"example"}'}}]}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'HB_API_BUDGET_ROOT':root,'HB_API_BUDGET_CNY':'25','DEEPSEEK_API_KEY':'fake-private-key'}):
            client=DeepSeek('dpo')
            with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(body).encode())) as request:
                output=client.generate([{'role':'user','content':'Return JSON.'}],512,1.)
            payload=json.loads(request.call_args.args[0].data)
            self.assertEqual(payload['thinking'],{'type':'disabled'})
            self.assertEqual(payload['response_format'],{'type':'json_object'})
            self.assertEqual(json.loads(output)['prompt'],'example')
            self.assertNotIn('fake-private-key',(Path(root)/'usage.jsonl').read_text())
            state=client.ledger.snapshot();self.assertFalse(state['pending']);self.assertEqual(state['settled_requests'],1)

    def test_api_runtime_keeps_official_pair_search_and_detects_tampering(self):
        import py_compile
        from fineval_pipeline.core import load_config,digest
        from fineval_pipeline.harmbench_pair_api import build_api_runtime
        project=Path(__file__).resolve().parents[1]
        if not (project/'third_party/HarmBench/.git').is_dir():
            self.skipTest('Run python3 finsec.py setup-harmbench for the upstream integration test')
        cfg=load_config(project/'configs/harmbench_gcg.example.yaml')
        with tempfile.TemporaryDirectory() as root:
            cfg['harmbench']['runtime_dir']=str(Path(root)/'canonical')
            cfg['harmbench']['api_runtime_dir']=str(Path(root)/'api')
            runtime=build_api_runtime(cfg)
            record=json.loads((runtime/'api_compatibility_manifest.json').read_text())
            canonical=Path(cfg['harmbench']['runtime_dir'])
            for file in record['unchanged_algorithms']:
                self.assertEqual(digest(runtime/file),digest(canonical/file))
            for row in record['changed_files']:py_compile.compile(str(runtime/row['file']),doraise=True)
            self.assertEqual(build_api_runtime(cfg),runtime)
            path=runtime/'baselines/pair/conversers.py';path.write_text(path.read_text()+'\n# tamper\n')
            with self.assertRaises(ValueError):build_api_runtime(cfg)


if __name__=='__main__':unittest.main()
