import unittest

from fineval_pipeline.privacy_exposure import training_nonmember_overlap


class PrivacyExposureTests(unittest.TestCase):
    def sample(self):
        return {'messages': [{'role': 'user', 'content': 'held-out financial problem'},
                             {'role': 'assistant', 'content': 'a held-out answer'}]}

    def test_sft_full_messages_and_normalized_prompt_are_both_checked(self):
        heldout = self.sample()
        result = training_nonmember_overlap([heldout], [heldout], 'sft')
        self.assertEqual(result, {'exact_message_overlap': 1, 'user_prompt_overlap': 1})
        safe = {'messages': [{'role': 'user', 'content': 'independent training question'},
                             {'role': 'assistant', 'content': 'a training answer'}]}
        self.assertFalse(any(training_nonmember_overlap([safe], [heldout], 'sft').values()))

    def test_a_dpo_rejected_response_cannot_bypass_nonmember_audit(self):
        heldout = self.sample()
        training = {'messages': heldout['messages'][:-1] + [{'role': 'assistant', 'content': 'another chosen answer'}],
                    'rejected_response': 'a held-out answer'}
        result = training_nonmember_overlap([training], [heldout], 'dpo')
        self.assertEqual(result['exact_message_overlap'], 0)
        self.assertEqual(result['user_prompt_overlap'], 1)
        self.assertEqual(result['rejected_message_overlap'], 1)

    def test_missing_dpo_branch_or_unknown_algorithm_is_not_silently_ignored(self):
        with self.assertRaises(ValueError): training_nonmember_overlap([self.sample()], [self.sample()], 'dpo')
        with self.assertRaises(ValueError): training_nonmember_overlap([], [], 'grpo')


if __name__ == '__main__':
    unittest.main()
