import unittest
from pi_crypto_agent.native_worker import tool_result


class LocalToolTests(unittest.TestCase):
    def setUp(self):
        self.public={'file':'example.py','code':'import hashlib\nx = hashlib.md5(data).hexdigest()\n',
            'knowledge':[{'id':'WEAK_HASH','rule':'Check security context'}]}

    def test_search_reports_actual_source_lines(self):
        result=tool_result('search_code',{'query':'md5'},self.public)
        self.assertEqual(result['matches'],[{'file':'example.py','line':2,'text':'x = hashlib.md5(data).hexdigest()'}])

    def test_tools_return_only_current_public_source(self):
        result=tool_result('read_source',{},self.public)
        self.assertEqual(set(result),{'file','source_with_lines'})
        self.assertNotIn('expected',result)
        self.assertEqual(tool_result('lookup_rules',{'rule_ids':['UNKNOWN']},self.public),{'rules':[]})

    def test_arbitrary_execution_and_unbounded_queries_rejected(self):
        with self.assertRaises(ValueError):tool_result('shell',{'cmd':'anything'},self.public)
        with self.assertRaises(ValueError):tool_result('search_code',{'query':'x'*101},self.public)
