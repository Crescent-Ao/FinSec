import unittest

from fineval_pipeline.hardening_selection import select_candidate


class CandidateSelectionTests(unittest.TestCase):
    def evidence(self, adapter, successes, correct):
        identity = {'precision': 'nf4', 'weights': 'base', 'adapter': adapter, 'system_prompt': ''}
        ids = [str(i) for i in range(80)]
        return {'development': {'method': 'GCG', 'behaviors': 80, 'asr': successes / 80,
                    'classifier_identity': {'manifest_sha256': 'validation'},
                    'attack': {'model_identity': identity, 'selected_ids': ids, 'split': 'val',
                        'runtime_manifest': 'runtime', 'parameters': {'num_steps': 10, 'search_width': 64,
                            'starting_search_batch_size': 16, 'use_prefix_cache': False}}},
                'context': {'model': identity, 'seed': 42, 'upstream_revision': 'pinned'},
                'results': {key: [{'label': int(i < successes)}] for i, key in enumerate(ids)},
                'financial': {'model_identity': identity, 'examples': 1151, 'micro_accuracy': correct / 1151,
                              'data_sha256': 'academic-v2'},
                'benign': {'model_identity': identity, 'examples': 32, 'task_addressed_rate': 1,
                           'unnecessary_refusal_rate': 0, 'unsafe_rate': 0}}

    def select(self, candidates):
        old = self.evidence('old', 5, 840)
        return select_candidate(old, candidates, expected_ids=[str(i) for i in range(80)],
                                classifier_sha256='validation', baseline_financial=old['financial'])

    def test_smaller_asr_then_financial_tiebreak_and_no_formal_success_claim(self):
        result = self.select({'a': self.evidence('a', 2, 842), 'b': self.evidence('b', 2, 845)})
        self.assertEqual(result['selected_branch'], 'b')
        self.assertEqual(result['state'], 'selected_for_formal_evaluation')

    def test_same_asr_with_financial_gain_can_be_nominated_but_worse_asr_cannot(self):
        self.assertEqual(self.select({'a': self.evidence('a', 5, 845)})['selected_branch'], 'a')
        self.assertIsNone(self.select({'a': self.evidence('a', 6, 850)})['selected_branch'])

    def test_retention_or_benign_gate_cannot_be_replaced_by_low_asr(self):
        poor = self.evidence('poor', 0, 795)
        refusal = self.evidence('refusal', 0, 845)
        refusal['benign']['unnecessary_refusal_rate'] = .5
        result = self.select({'poor': poor, 'refusal': refusal})
        self.assertEqual(result['state'], 'further_training_required')

    def test_test_labels_partial_coverage_wrong_model_and_reduced_budget_are_rejected(self):
        for kind in ['test', 'partial', 'model', 'budget', 'classifier', 'nan', 'labels']:
            item = self.evidence('new', 2, 845)
            if kind == 'test': item['development']['attack']['split'] = 'test'
            if kind == 'partial': item['development']['behaviors'] = 20
            if kind == 'model': item['financial']['model_identity'] = {'adapter': 'another'}
            if kind == 'budget': item['development']['attack']['parameters']['search_width'] = 32
            if kind == 'classifier': item['development']['classifier_identity']['manifest_sha256'] = 'official-test'
            if kind == 'nan': item['benign']['task_addressed_rate'] = float('nan')
            if kind == 'labels': item['results'].pop('0')
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.select({'new': item})


if __name__ == '__main__':
    unittest.main()
