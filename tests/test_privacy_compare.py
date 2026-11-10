import unittest

from fineval_pipeline.privacy_compare import align_scores, paired_changes


class PairedPrivacyTests(unittest.TestCase):
    def fixture(self):
        return [{'id': str(i), 'member': int(i >= 4), 'group_id': str(i // 2),
                 'answer_tokens': 10, 'loss_score': i / 10., 'min_k_score': i / 20.}
                for i in range(8)]

    def test_alignment_uses_ids_and_rejects_changed_membership(self):
        a = self.fixture()
        b = list(reversed(self.fixture()))
        aligned = align_scores({'M0': a, 'M2': b})
        self.assertEqual(aligned['M0'], aligned['M2'])
        b[0]['member'] = 0
        with self.assertRaisesRegex(ValueError, 'labels/families'):
            align_scores({'M0': a, 'M2': b})
        with self.assertRaisesRegex(ValueError, 'samples differ'):
            align_scores({'M0': a, 'M2': b[:-1]})

    def test_identical_models_have_exactly_zero_paired_change(self):
        result = paired_changes({'M0': self.fixture(), 'M1': self.fixture(), 'M2': self.fixture()},
                                [('safety', 'M1', 'M2')], repeats=20)
        for attack in result['comparisons']['safety']['attacks'].values():
            for metric in attack.values():
                self.assertEqual(metric['delta_after_minus_before'], 0.)
                self.assertEqual(metric['paired_family_bootstrap_95pct_ci'], [0., 0.])
                self.assertEqual(metric['direction'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()
