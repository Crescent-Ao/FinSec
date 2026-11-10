import json
from pathlib import Path
import tempfile
import unittest

from fineval_pipeline.core import digest
from fineval_pipeline.gcg_native_generation import verify_adoption


class NativePoolAdoptionTests(unittest.TestCase):
    def fixture(self, root):
        hard = root / 'outputs/security_hardening_20261109_v1'
        proof = hard / 'GCG_completion_replication_full320'
        proof.mkdir(parents=True)
        runtime = root / 'runtime/native'
        runtime.mkdir(parents=True)
        (root / 'scripts').mkdir()
        files = [(root / 'scripts/verify_gcg_completion_replication.py', 'pool_source_sha256'),
                 (root / 'scripts/probe_gcg_completion_replicas.py', 'native_worker_sha256'),
                 (runtime / 'generate_completions.py', 'native_source_sha256')]
        for path, _ in files:
            path.write_text(path.name)
        plan = {key: digest(path) for path, key in files}
        (proof / 'plan.json').write_text(json.dumps(plan))
        baseline = hard / 'formal_finance_scoring/evaluation/M1/harmbench/GCG/completions.json'
        baseline.parent.mkdir(parents=True)
        baseline.write_text(json.dumps({str(i): [{'generation': 'synthetic answer'}] for i in range(320)}))
        (proof / 'completions.json').write_bytes(baseline.read_bytes())
        identity = {'weights': 'base', 'adapter': 'candidate', 'precision': 'nf4'}
        result = {'behaviors': 320, 'records_identical_to_reference': 320,
            'all320_case_and_answer_records_identical_to_reference': True,
            'whole_completion_file_byte_identical_to_reference': True,
            'source_plan_sha256': digest(proof / 'plan.json'),
            'completions_sha256': digest(baseline),
            'model_identity': {'weights': 'base', 'adapter': 'baseline', 'precision': 'nf4'}}
        (proof / 'result.json').write_text(json.dumps(result))
        policy = dict(plan, enabled=True, allowed_model_identities=[identity],
            proof_result_sha256=digest(proof / 'result.json'), proof_plan_sha256=digest(proof / 'plan.json'))
        (hard / 'native_pool_adoption.json').write_text(json.dumps(policy))
        return hard, proof, runtime, identity, baseline

    def test_complete_matching_proof_covers_exact_registered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hard, proof, runtime, identity, _ = self.fixture(root)
            result = verify_adoption(root, identity, runtime)
            self.assertEqual(result['full320_result_sha256'], digest(proof / 'result.json'))

    def test_changed_code_or_proof_or_reference_is_rejected(self):
        for mode in ['source', 'proof', 'reference']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                hard, proof, runtime, identity, baseline = self.fixture(root)
                path = {'source': runtime / 'generate_completions.py',
                        'proof': proof / 'result.json', 'reference': baseline}[mode]
                path.write_text('{}')
                with self.assertRaises(ValueError): verify_adoption(root, identity, runtime)

    def test_partial_claim_is_rejected_even_with_matching_saved_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hard, proof, runtime, identity, _ = self.fixture(root)
            result = json.loads((proof / 'result.json').read_text())
            result['records_identical_to_reference'] = 319
            (proof / 'result.json').write_text(json.dumps(result))
            policy = json.loads((hard / 'native_pool_adoption.json').read_text())
            policy['proof_result_sha256'] = digest(proof / 'result.json')
            (hard / 'native_pool_adoption.json').write_text(json.dumps(policy))
            with self.assertRaises(ValueError): verify_adoption(root, identity, runtime)

    def test_other_checkpoint_or_bf16_precision_is_not_covered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, runtime, identity, _ = self.fixture(root)
            for changed in [dict(identity, adapter='unregistered'), dict(identity, precision='bf16')]:
                with self.assertRaises(ValueError): verify_adoption(root, changed, runtime)


if __name__ == '__main__':
    unittest.main()
