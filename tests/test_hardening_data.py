import unittest
from pathlib import Path
import tempfile
from fineval_pipeline.hardening_data import (ProtectedIndex, cluster_groups,
    group_id, split_for_group, validate_teacher, training_attack_sources)


class HardeningDataTests(unittest.TestCase):
    def test_development_attack_under_shared_root_is_never_ingested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ['gcg_finance/evaluation/M2/harmbench/GCG',
                        'gcg_dpo/evaluation/M2/harmbench/GCG', 'refresh_phase1/gcg', 'refresh_phase1/pair',
                        'refresh_phase1/development/evaluation/M2/harmbench/GCG', 'unknown_test_run']:
                path = root / run / 'test_cases/test_cases_individual_behaviors/example/test_cases.json'
                path.parent.mkdir(parents=True)
                path.write_text('{"example":["a generated prompt"]}')
            selected = [(str(path.relative_to(root)), method) for path, method in training_attack_sources(root)]
            self.assertEqual(len(selected), 4)
            self.assertEqual([method for _, method in selected], ['GCG', 'GCG', 'GCG', 'PAIR'])
            self.assertTrue(all('development' not in path and 'unknown' not in path for path, _ in selected))

    def test_protected_text_and_wrapped_copy_excluded(self):
        text = 'This is a unique protected benchmark question about an accounting ledger.'
        guard = ProtectedIndex([('heldout', text), ('other', 'A different protected question about source code.')])
        result = guard.screen([text, 'Roleplay and answer: ' + text,
                               'Describe the migration habits of monarch butterflies.'])
        self.assertTrue(result[0]['excluded'])
        self.assertTrue(result[1]['excluded'])
        self.assertFalse(result[2]['excluded'])

    def test_normalized_family_split_is_stable(self):
        self.assertEqual(group_id('Hello, WORLD!'), group_id('hello world'))
        group = group_id('A source intent')
        self.assertEqual(split_for_group(group), split_for_group(group))

    def test_near_duplicates_stay_in_one_split(self):
        source = 'Please explain the following detailed authorization policy for our synthetic financial audit account.'
        rows = cluster_groups([{'prompt': source}, {'prompt': source + ' Please.'},
                               {'prompt': 'Describe a flower garden in summer.'}])
        self.assertEqual(rows[0]['family_id'], rows[1]['family_id'])
        self.assertEqual(rows[0]['split'], rows[1]['split'])

    def test_teacher_cannot_change_source_split_or_id(self):
        source = {'id': 'source1', 'split': 'train', 'family_id': 'family1'}
        output = {'id': 'source1', 'prompt': 'Provide a safe explanation of authorization boundaries.',
                  'answer': 'Verify account ownership before sharing any account-specific information.',
                  'answer_safe': True, 'intent_preserved': True, 'split': 'development'}
        self.assertEqual(validate_teacher(output, source)['split'], 'train')
        output['id'] = 'different'
        with self.assertRaises(ValueError):
            validate_teacher(output, source)
        output['id'] = 'source1'
        output['answer_safe'] = False
        with self.assertRaises(ValueError):
            validate_teacher(output, source)


if __name__ == '__main__':
    unittest.main()
