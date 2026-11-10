"""DDP-safe exposure, optimizer-resume and synchronization audit."""
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from swift.callbacks import TrainerCallback, callbacks_map


def signature(model, gradients=False):
    h = hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        if not p.requires_grad: continue
        value = p.grad if gradients else p
        if value is None: continue
        x = value.detach().flatten()[:256].float().cpu().contiguous()
        if not torch.isfinite(x).all(): raise FloatingPointError(name)
        h.update(name.encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


class AcceleratedAudit(TrainerCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        self.run=Path(os.environ['DISTILLED_RUN']);m=json.loads((self.run/'manifest.json').read_text())
        self.world=dist.get_world_size() if dist.is_initialized() else 1;self.rank=dist.get_rank() if self.world>1 else 0
        assert len(self.trainer.train_dataset)==m['splits']['train']['rows']
        assert len(self.trainer.eval_dataset)==m['splits']['validation']['rows']
        assert self.world*args.per_device_train_batch_size*args.gradient_accumulation_steps==16
        assert state.max_steps==math.ceil(m['splits']['train']['rows']/16)
        self.initial_step=state.global_step;self.initial_hash=signature(self.trainer.model)
        saved_state=json.loads((Path(os.environ['ACCELERATED_RESUME'])/'trainer_state.json').read_text())
        assert self.initial_step==saved_state['global_step'],'Trainer progress was reset instead of restored'
        from peft import get_peft_model_state_dict
        from safetensors import safe_open
        base=self.trainer.model.module if hasattr(self.trainer.model,'module') else self.trainer.model
        adapters=get_peft_model_state_dict(base)
        with safe_open(str(Path(os.environ['ACCELERATED_RESUME'])/'adapter_model.safetensors'),framework='pt',device='cpu') as checkpoint:
            assert set(adapters)==set(checkpoint.keys()),'Adapter checkpoint keys differ'
            for name,value in adapters.items():
                expected=checkpoint.get_slice(name)
                # Adapter matrices are 2D; read a bounded row sample from each.
                n=min(2,value.shape[0]);saved=expected[:n].float()
                assert torch.equal(value[:n].detach().float().cpu(),saved),'Adapter weights were not restored: '+name
        optimizer=self.trainer.optimizer
        steps=[int(v['step']) for v in optimizer.state.values() if 'step' in v]
        if self.initial_step:
            assert steps and min(steps)==max(steps)==self.initial_step, 'Optimizer state was not restored'
        self.info={'world_size':self.world,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                   'micro_batch':args.per_device_train_batch_size,'gradient_accumulation_steps':args.gradient_accumulation_steps,
                   'effective_batch':16,'initial_global_step':self.initial_step,'max_steps':state.max_steps,
                   'optimizer_state_step_min':min(steps) if steps else None,'optimizer_state_step_max':max(steps) if steps else None,
                   'restored_learning_rates':[g['lr'] for g in optimizer.param_groups],
                   'checkpoint_adapter_samples_matched':True,
                   'kernels':'FLA + causal-conv1d + Liger fused linear CE only',
                   'train_rows':len(self.trainer.train_dataset),'validation_rows':len(self.trainer.eval_dataset)}
        torch.cuda.reset_peak_memory_stats()
        if self.rank==0:(self.run/'accelerated_runtime.json').write_text(json.dumps(self.info,indent=2))

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        if self.world>1 and state.global_step==self.initial_step:
            digests=[None]*self.world;dist.all_gather_object(digests,signature(self.trainer.model,True))
            assert len(set(digests))==1,'DDP gradient sample mismatch'

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step>=state.max_steps:control.should_evaluate=True;control.should_save=True
        return control

    def on_log(self,args,state,control,logs=None,**kwargs):
        for name in ('loss','grad_norm','eval_loss'):
            if name in (logs or {}) and not math.isfinite(float(logs[name])):raise FloatingPointError(name)

    def on_save(self,args,state,control,**kwargs):
        low=torch.tensor(int(shutil.disk_usage(args.output_dir).free<20*1024**3),device=args.device)
        if self.world>1:dist.all_reduce(low,op=dist.ReduceOp.MAX)
        if low.item():raise RuntimeError('Checkpoint saved; insufficient free space to continue')

    def on_train_end(self,args,state,control,**kwargs):
        final=signature(self.trainer.model)
        assert final!=self.initial_hash
        info={'rank':self.rank,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
              'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3,'parameter_sample_sha256':final}
        ranks=[info]
        if self.world>1:ranks=[None]*self.world;dist.all_gather_object(ranks,info)
        assert len({r['parameter_sample_sha256'] for r in ranks})==1
        if self.rank==0:
            report=self.info|{'probe':False,'global_step':state.global_step,'epoch':state.epoch,
                 'ranks':ranks,'sampled_gradient_and_weight_sync':'verified' if self.world>1 else 'single_gpu',
                 'peak_allocated_GiB':max(r['peak_allocated_GiB'] for r in ranks),
                 'peak_reserved_GiB':max(r['peak_reserved_GiB'] for r in ranks)}
            (self.run/'training/audit.json').write_text(json.dumps(report,indent=2))


callbacks_map['accelerated_audit']=AcceleratedAudit
