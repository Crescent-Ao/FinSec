import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.modelscope_finsec import validate_adapter


class ModelScopeReleaseTests(unittest.TestCase):
    def test_adapter_replacement_is_detected_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight, config = b'authored-test-weights', b'{}'
            (root / 'adapter_model.safetensors').write_bytes(weight)
            (root / 'adapter_config.json').write_bytes(config)
            manifest = {'format': 'unmerged_peft_lora', 'loading': {'quantization': 'bitsandbytes_nf4'},
                'adapter': {'file': 'adapter_model.safetensors', 'sha256': hashlib.sha256(weight).hexdigest(),
                            'config_file': 'adapter_config.json', 'config_sha256': hashlib.sha256(config).hexdigest()}}
            (root / 'finsec_release.json').write_text(json.dumps(manifest))
            self.assertEqual(validate_adapter(root), manifest)
            (root / 'adapter_model.safetensors').write_bytes(b'wrong-adapter')
            with self.assertRaisesRegex(ValueError, 'integrity check failed'):
                validate_adapter(root)


if __name__ == '__main__':
    unittest.main()
