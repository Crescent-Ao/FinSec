import unittest

from fineval_pipeline.agent import parse_findings
from scripts.stage2_verifiers import strict_json, verify


class RewardSerializationTests(unittest.TestCase):
    def setUp(self):
        self.task = {'task_type': 'crypto_detection', 'private_ground_truth': {
            'files': [{'path': 'a.py', 'code': 'x=1\n'}], 'findings': []}}

    def test_empty_thinking_scaffold_keeps_correct_reward(self):
        result = verify(self.task, {'completion': '<think>\n\n</think>\n\n{"findings":[]}'})
        self.assertTrue(result['valid'])
        self.assertEqual(result['reward'], 1)

    def test_nonempty_thought_is_still_a_format_failure(self):
        result = verify(self.task, {'completion': '<think>reasoning</think>{"findings":[]}'})
        self.assertFalse(result['valid'])
        self.assertEqual(result['reward'], -1)

    def test_duplicate_json_keys_are_still_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('<think></think>{"findings":[],"findings":[]}')

    def test_evidence_mismatch_is_still_rejected(self):
        candidate = '<think></think>{"findings":[{"file":"a.py","line":1,' \
                    '"rule_id":"WEAK_HASH","evidence":"fabricated","reason":"r","fix":"f"}]}'
        self.assertEqual(verify(self.task, {'completion': candidate})['reward'], -1)

    def test_audit_accepts_the_same_empty_scaffold(self):
        self.assertEqual(parse_findings('<think>\n</think>\n{"findings":[]}', 'a.py', 'x=1', set()), [])


if __name__ == '__main__':
    unittest.main()
