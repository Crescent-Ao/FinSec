"""Record consumed dataset size and stage completion for the experiment chain."""
import json
import os
from pathlib import Path
import torch
from swift.callbacks import TrainerCallback, callbacks_map


class Stage2Audit(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.run = Path(os.environ['STAGE2_RUN'])
        self.stage = os.environ['STAGE2_STAGE']
        manifest = json.loads((self.run / 'data_manifest.json').read_text())
        expected = manifest['files'][self.stage + '_train.jsonl']['rows']
        actual = len(self.trainer.train_dataset)
        if actual != expected:
            raise RuntimeError(f'Training preprocessing retained {actual} of {expected} frozen rows')
        model = self.trainer.model
        if not getattr(model, 'is_loaded_in_4bit', False):
            raise RuntimeError('The active stage-2 protocol requires a 4-bit NF4 base')
        quantization = model.config.quantization_config.to_dict()
        if quantization['bnb_4bit_quant_type'] != 'nf4' or not quantization['bnb_4bit_use_double_quant']:
            raise RuntimeError('The loaded quantization configuration differs from the NF4 baseline')
        if quantization['bnb_4bit_compute_dtype'] != 'bfloat16':
            raise RuntimeError('NF4 baseline requires BF16 compute')
        self.record = {'stage': self.stage, 'training_rows': actual,
                       'quantization': quantization,
                       'parent_checkpoint': os.environ['STAGE2_PARENT'],
                       'initial_step': state.global_step,
                       'effective_batch': args.per_device_train_batch_size * args.gradient_accumulation_steps
                                          * int(os.environ.get('WORLD_SIZE', '1'))}
        (self.run / (self.stage + '_audit.json')).write_text(json.dumps(self.record, indent=2) + '\n')

    def on_train_end(self, args, state, control, **kwargs):
        self.record.update(global_step=state.global_step, epoch=state.epoch,
                           peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
                           peak_reserved_GiB=torch.cuda.max_memory_reserved() / 1024**3,
                           final_logs=state.log_history[-5:])
        (self.run / (self.stage + '_audit.json')).write_text(json.dumps(self.record, indent=2) + '\n')


callbacks_map['stage2_audit'] = Stage2Audit
