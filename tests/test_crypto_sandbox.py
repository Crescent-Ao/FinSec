import unittest
from pi_crypto_agent.sandbox import run_python_test


class CryptoExecutionTests(unittest.TestCase):
    public={'file':'payment.py','code':'def add(a,b):\n    return a+b\n'}

    def test_executes_actual_source_and_assertions(self):
        r=run_python_test({'test_code':'from audited_source import add\nassert add(2,3)==5\nprint("tested")'},self.public)
        self.assertTrue(r['test_passed'],r);self.assertIn('tested',r['stdout'])

    def test_failure_is_not_a_pass(self):
        r=run_python_test({'test_code':'assert False, "intentional assertion"'},self.public)
        self.assertFalse(r['test_passed']);self.assertNotEqual(r['exit_code'],0)

    def test_host_workspace_and_credentials_unavailable(self):
        code='import os\nassert not os.path.exists("/home/ps/qwen_lora")\nassert "DEEPSEEK_API_KEY" not in os.environ\nassert not os.path.exists("/dev/nvidia0")\nprint("isolated")'
        r=run_python_test({'test_code':code},self.public);self.assertTrue(r['test_passed'],r)

    def test_real_crypto_property_and_candidate_fix(self):
        source='from Crypto.Cipher import AES\ndef encrypt(data):\n    return AES.new(b"k"*16,AES.MODE_ECB).encrypt(data)\n'
        test='from audited_source import encrypt\nc=encrypt(b"a"*32)\nassert c[:16]==c[16:]\nprint("ECB repeats identical blocks")'
        r=run_python_test({'test_code':test},{'file':'crypto.py','code':source});self.assertTrue(r['test_passed'],r)
        r=run_python_test({'candidate_code':'def result():\n    return 7\n','test_code':'from audited_source import result\nassert result()==7'},self.public)
        self.assertTrue(r['test_passed'],r);self.assertEqual(r['source_kind'],'candidate_fix')

    def test_other_languages_are_not_falsely_marked_passed(self):
        r=run_python_test({'test_code':'print(1)'},{'file':'service.java','code':'class Service {}'})
        self.assertEqual(r['status'],'unsupported_language');self.assertNotIn('test_passed',r)
