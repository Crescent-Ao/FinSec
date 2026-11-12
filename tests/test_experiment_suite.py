import json
from pathlib import Path
import tempfile
import unittest

from scripts.experiment_suite import collect, prepare

ROOT = Path(__file__).resolve().parents[1]


class ExperimentSuiteTests(unittest.TestCase):
    def config(self, directory):
        config = json.loads((ROOT / 'configs/evaluation_suite.example.json').read_text())
        config['output_root'] = str(Path(directory) / 'outputs')
        path = Path(directory) / 'suite.json'
        path.write_text(json.dumps(config))
        return path, config

    def test_five_stage_plan_preserves_base_and_distinct_adapters(self):
        import yaml
        with tempfile.TemporaryDirectory() as temp:
            path, config = self.config(temp)
            plan = prepare(path)
            self.assertEqual(len(plan['jobs']), 5)
            generated = yaml.safe_load((Path(plan['output_root']) / 'suite.yaml').read_text())
            self.assertNotIn('adapter', generated['models']['M0'])
            self.assertEqual(len({m['adapter'] for n, m in generated['models'].items() if n != 'M0'}), 4)
            self.assertEqual({m['precision'] for m in generated['models'].values()}, {'nf4'})
            self.assertFalse((Path(plan['output_root']) / 'status.json').exists())
            self.assertEqual(prepare(path), plan)
            old = (Path(plan['output_root']) / 'suite.yaml').read_text()
            config['gpus'] = '1';path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                prepare(path)
            self.assertEqual((Path(plan['output_root']) / 'suite.yaml').read_text(), old)

    def test_attack_plan_separates_search_and_official_judging(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(temp)
            plan = prepare(path, ['gcg', 'pair', 'agent'])
            self.assertEqual(len(plan['jobs']), 25)
            self.assertIn('attack', plan['jobs'][0]['command'])
            self.assertIn('judge', plan['jobs'][1]['command'])
            self.assertIn('evaluate_described_4096_pi_only.mjs', ' '.join(plan['jobs'][4]['command']))

    def test_missing_results_stay_missing_and_protocol_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(temp)
            plan = prepare(path)
            result = collect(plan)
            self.assertFalse(result['complete'])
            self.assertEqual(len(result['missing']), 5)
            for name, score in [('M0', .7), ('M1', .75)]:
                target = Path(plan['output_root']) / name / 'fineval/summary.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({'task': 'options', 'data_sha256': 'same-data', 'examples': 10,
                    'limit_per_subject': 0, 'micro_accuracy': score}))
            result = collect(plan)
            self.assertAlmostEqual(result['stages'][1]['fineval_delta_pp_vs_base'], 5)
            target = Path(plan['output_root']) / 'M1/fineval/summary.json'
            changed = json.loads(target.read_text());changed['examples'] = 9
            target.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'protocols differ'):
                collect(plan)

    def test_adapter_on_base_and_changed_plan_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, config = self.config(temp)
            config['stages'][0]['adapter'] = 'models/incorrect'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'unadapted baseline'):
                prepare(path)
            config['stages'][0]['adapter'] = None
            path.write_text(json.dumps(config));prepare(path)
            config['stages'][1]['adapter'] = 'models/different'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                prepare(path)


if __name__ == '__main__':
    unittest.main()
