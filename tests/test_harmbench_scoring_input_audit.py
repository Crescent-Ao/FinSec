import unittest

from scripts.audit_harmbench_scoring_inputs import verify_transformation


class FakeTokenizer:
    def encode(self, value, *, max_length, truncation):
        return list(value[:max_length])

    def decode(self, ids, *, skip_special_tokens):
        return ''.join(ids)


class OfficialScoringInputTests(unittest.TestCase):
    def fixture(self):
        original = {'a': [{'test_case': 'synthetic educational request', 'generation': 'ABCDEFGH'}]}
        labels = {'a': [{'test_case': 'synthetic educational request', 'generation': 'ABCD', 'label': 0}]}
        cases = {'a': ['synthetic educational request']}
        return original, labels, cases

    def test_exact_official_clipping_is_accepted_and_counted(self):
        result = verify_transformation(*self.fixture(), FakeTokenizer(), token_limit=4)
        self.assertEqual(result['behaviors_verified'], 1)
        self.assertEqual(result['generation_text_changed_by_official_tokenization'], 1)

    def test_different_generation_or_attack_case_is_not_accepted_as_clipping(self):
        for mode in ['generation', 'result_case', 'source_case', 'partial', 'multiple']:
            with self.subTest(mode=mode):
                original, labels, cases = self.fixture()
                if mode == 'generation': labels['a'][0]['generation'] = 'ABCE'
                if mode == 'result_case': labels['a'][0]['test_case'] = 'different case'
                if mode == 'source_case': cases['a'][0] = 'different case'
                if mode == 'partial': labels.clear()
                if mode == 'multiple': original['a'].append(original['a'][0])
                with self.assertRaises(ValueError):
                    verify_transformation(original, labels, cases, FakeTokenizer(), token_limit=4)


if __name__ == '__main__':
    unittest.main()
