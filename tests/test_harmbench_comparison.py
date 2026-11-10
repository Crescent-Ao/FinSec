import unittest

from fineval_pipeline.harmbench_comparison import paired_label_comparison, wilson_interval


class HarmBenchComparisonTests(unittest.TestCase):
    def rows(self, values):
        return {str(i): [{'label': value}] for i, value in enumerate(values)}

    def test_identical_results_have_zero_paired_change_and_no_evidence_of_difference(self):
        rows = self.rows([1, 0, 0, 1])
        result = paired_label_comparison(rows, rows, rows)
        self.assertEqual(result['paired_behavior_bootstrap95'], [0., 0.])
        self.assertEqual(result['exact_McNemar_two_sided_p'], 1)

    def test_equal_aggregate_scores_preserve_improved_and_regressed_behaviors(self):
        before, after = self.rows([1, 1, 0, 0]), self.rows([1, 0, 1, 0])
        result = paired_label_comparison(before, after, before)
        self.assertEqual((result['paired_improved'], result['paired_regressed']), (1, 1))
        self.assertEqual(result['ASR_change'], 0)
        self.assertLess(result['paired_behavior_bootstrap95'][0], 0)
        self.assertGreater(result['paired_behavior_bootstrap95'][1], 0)

    def test_all_improved_has_exact_sign_test_and_finite_bounds(self):
        before, after = self.rows([1] * 8), self.rows([0] * 8)
        result = paired_label_comparison(before, after, before)
        self.assertEqual(result['exact_McNemar_two_sided_p'], 2 / 256)
        self.assertEqual(result['ASR_change'], -1)
        for interval in [wilson_interval(0, 8), wilson_interval(8, 8)]:
            self.assertTrue(0 <= interval[0] <= interval[1] <= 1)

    def test_partial_or_multi_case_labels_do_not_silently_improve_the_score(self):
        before, after = self.rows([1, 0]), self.rows([0])
        with self.assertRaises(ValueError): paired_label_comparison(before, after, before)
        after = self.rows([0, 0]);after['0'].append({'label': 0})
        with self.assertRaises(ValueError): paired_label_comparison(before, after, before)
        after = self.rows([0, '0'])
        with self.assertRaises(ValueError): paired_label_comparison(before, after, before)


if __name__ == '__main__':
    unittest.main()
