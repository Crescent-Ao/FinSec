import unittest

from fineval_pipeline.evaluation_exposure import ExactFinancialReferenceIndex
from scripts.validate_pilot_data import normalized


class ExactExposureTests(unittest.TestCase):
    def setUp(self):
        self.row = {'subject': 'finance', 'id': 'short', 'question': '哪一项属于这里的权益资产',
                    'options': {'A': '普通股票', 'B': '银行存款', 'C': '政府债券', 'D': '现金余额'}}
        self.index = ExactFinancialReferenceIndex([self.row])

    def test_short_reference_can_be_found_below_old40_character_minimum(self):
        text = self.row['question'] + ''.join(self.row['options'].values())
        self.assertLess(len(normalized(text)), 40)
        self.assertEqual(self.index.search(normalized(text))[0]['reference'], 'finance:short')

    def test_standard_lettered_choices_and_surrounding_text_are_detected(self):
        text = self.row['question'] + ' '.join(k + '. ' + self.row['options'][k] for k in 'ABCD')
        result = self.index.search(normalized('下面是一条记录：' + text + '。答案在解释中。'))
        self.assertTrue(any(r['kind'] == 'with_dotted_options' for r in result))

    def test_english_without_any_matching_reference_is_not_flagged(self):
        self.assertFalse(self.index.search(normalized('This independent training record has no matching reference.')))


if __name__ == '__main__':
    unittest.main()
