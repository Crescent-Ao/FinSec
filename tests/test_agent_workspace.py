import unittest
from pi_crypto_agent.workspace_tools import workspace_tool


class WorkspaceToolsTests(unittest.TestCase):
    public={'file':'test_workspace.py','code':'def encrypt(x):\n    return x\n'}

    def test_write_run_and_modify_candidate(self):
        workspace_tool('write_file',{'path':'test_case.py','content':'from audited_source import encrypt\nassert encrypt(7)==7\nprint("tested original")'},self.public)
        r=workspace_tool('run_command',{'command':'python3.11 test_case.py'},self.public)
        self.assertEqual(r['exit_code'],0,r);self.assertIn('tested original',r['stdout'])
        workspace_tool('replace_text',{'path':'audited_source.py','old':'return x','new':'return x + 1'},self.public)
        r=workspace_tool('run_command',{'command':'python3.11 test_case.py'},self.public)
        self.assertNotEqual(r['exit_code'],0)

    def test_shell_and_crypto_dependency(self):
        r=workspace_tool('run_command',{'command':"python3.11 -c 'from Crypto.Cipher import AES; print(AES.block_size)'"},self.public)
        self.assertEqual(r['exit_code'],0,r);self.assertIn('16',r['stdout'])

    def test_escape_and_other_case_isolation(self):
        with self.assertRaises(ValueError):workspace_tool('read_file',{'path':'../../etc/passwd'},self.public)
        other={'file':'other.py','code':'pass\n'}
        self.assertEqual(workspace_tool('list_files',{},other)['files'],['audited_source.py'])
        r=workspace_tool('run_command',{'command':'test ! -e /home/ps/qwen_lora && test ! -e /dev/nvidia0 && echo isolated'},other)
        self.assertEqual(r['exit_code'],0,r)
