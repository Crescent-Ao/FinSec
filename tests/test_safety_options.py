import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fineval_pipeline.safety import build_plan


class SafetyOptionsTests(unittest.TestCase):
    def test_legacy_defaults_and_prepared_nf4_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'validation', 'private'):
                record = {'id': split, 'group_id': split, 'source': 'authored',
                          'messages': [{'role': 'user', 'content': split},
                                       {'role': 'assistant', 'content': 'response '+split}]}
                (root/(split+'.jsonl')).write_text(json.dumps(record)+'\n')
            (root/'cases.jsonl').write_text('')
            with (root/'behaviors.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['BehaviorID', 'Behavior', 'ContextString'])
                writer.writerow(['one', 'distinct', ''])
            cfg = {'seed': 42, 'safety_training': {'train': str(root/'train.jsonl'),
                   'validation': str(root/'validation.jsonl'), 'output': str(root/'default'),
                   'epochs': 1, 'learning_rate': 1e-5},
                   'privacy': {'nonmembers': str(root/'private.jsonl')},
                   'fineval': {'data_dir': str(root/'val')},
                   'harmbench': {'behaviors': str(root/'behaviors.csv')},
                   'agent': {'cases': str(root/'cases.jsonl')}}
            guard = Mock()
            guard.search.return_value = []
            parent = {'path': str(root/'base'), 'adapter': str(root/'adapter')}
            with patch('fineval_pipeline.safety.model_spec', return_value=parent), \
                 patch('fineval_pipeline.safety.model_identity', return_value={'test': True}), \
                 patch('scripts.validate_pilot_data.load_fineval', return_value=(guard, 4661)):
                default = build_plan(cfg)['command']
                self.assertEqual(default[default.index('--max_length')+1], '2048')
                self.assertNotIn('--quant_bits', default)
                configured = copy.deepcopy(cfg)
                configured['safety_training'].update(output=str(root/'nf4'), max_length=4096,
                    quant_bits=4, lora_rank=16, lora_alpha=32,
                    per_device_train_batch_size=1, gradient_accumulation_steps=16)
                command = build_plan(configured)['command']
                for flag, value in {'--max_length': '4096', '--quant_bits': '4',
                                    '--lora_rank': '16', '--lora_alpha': '32',
                                    '--per_device_train_batch_size': '1',
                                    '--gradient_accumulation_steps': '16',
                                    '--resume_only_model': 'true'}.items():
                    self.assertEqual(command[command.index(flag)+1], value)
                invalid = copy.deepcopy(cfg)
                invalid['safety_training'].update(output=str(root/'invalid'), quant_bits=8)
                with self.assertRaises(ValueError):
                    build_plan(invalid)
                self.assertFalse((root/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
