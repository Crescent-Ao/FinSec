"""Full-corpus sampler, DDP synchronization and one-epoch training audit."""
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from swift.callbacks import TrainerCallback, callbacks_map
from swift.dataloader.shard import BatchSamplerShard
import swift.trainers.mixin as trainer_mixin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.accelerated_training_audit import signature


class CompleteBatchSampler(BatchSamplerShard):
    """Cover every row and pad at most world_size-1 rows for equal DDP lengths.

    Swift's installed random sampler rounds the corpus down to a multiple of
    world size. Its length grouping can instead give ranks unequal lengths.
    This sampler always uses the full corpus and equalizes the rank lengths.
    """
    def __init__(self,total_samples,*args,**kwargs):
        super().__init__(total_samples,*args,**kwargs)
        self.real_samples=total_samples
        self.total_samples=math.ceil(total_samples/self.world_size)
        self.reference_batch_size=int(os.environ.get('LENGTH_GROUP_REFERENCE_BATCH_SIZE',str(self.batch_size*self.world_size)))

    def __iter__(self):
        generator=torch.Generator().manual_seed(self.curr_seed)
        if self.shuffle and self.group_by_length:
            from transformers.trainer_pt_utils import get_length_grouped_indices
            indices=get_length_grouped_indices(self.lengths,self.reference_batch_size,generator=generator)
        elif self.shuffle:indices=torch.randperm(self.real_samples,generator=generator).tolist()
        else:indices=list(range(self.real_samples))
        padding=self.total_samples*self.world_size-len(indices)
        if padding:indices += [indices[i%len(indices)] for i in range(padding)]
        local=indices[self.rank::self.world_size]
        for start in range(0,len(local),self.batch_size):
            batch=local[start:start+self.batch_size]
            if len(batch)==self.batch_size or not self.drop_last:yield batch


# Patch only this isolated training process, before Trainer builds its loader.
trainer_mixin.BatchSamplerShard=CompleteBatchSampler


class MergedAudit(TrainerCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        self.run=Path(os.environ['MERGED_RUN']);manifest=json.loads((self.run/'manifest.json').read_text())
        self.world=dist.get_world_size() if dist.is_initialized() else 1
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.initial_step=state.global_step
        self.initial_hash=signature(self.trainer.model)
        from scripts.activation_checkpoint_policy import retain_layer_activations
        retained_layers=retain_layer_activations(self.trainer.model,int(os.environ.get('RETAIN_ACTIVATION_LAYERS','0')))
        quantized=bool(getattr(self.trainer.model,'is_quantized',False))
        expected_quantized=os.environ.get('EXPECTED_QUANTIZED_BASE')
        if expected_quantized is not None:
            assert quantized==(expected_quantized=='1'),'Actual base quantization differs from requested method'
        n=manifest['splits']['train']['rows'];assert len(self.trainer.train_dataset)==n
        assert len(self.trainer.eval_dataset)==manifest['splits']['validation']['rows']
        effective=self.world*args.per_device_train_batch_size*args.gradient_accumulation_steps
        assert self.world==int(os.environ.get('EXPECTED_WORLD_SIZE','3')) and effective==24
        assert state.max_steps==math.ceil(n/effective)
        loader=kwargs['train_dataloader'];sampler=loader.batch_sampler
        while not isinstance(sampler,CompleteBatchSampler) and hasattr(sampler,'batch_sampler'):sampler=sampler.batch_sampler
        assert isinstance(sampler,CompleteBatchSampler),'Complete-corpus sampler was not installed'
        # Verify actual sampler indices without loading examples or invoking CUDA.
        seen=np.zeros(n,dtype=np.uint8);count=0;batches=0
        for indices in sampler:
            assert all(0<=i<n for i in indices)
            seen[indices]=1;count+=len(indices);batches+=1
        local={'seen':seen.tobytes(),'count':count,'batches':batches}
        rows=[local]
        if self.world>1:
            rows=[None]*self.world;dist.all_gather_object(rows,local)
        union=np.zeros(n,dtype=np.uint8)
        for row in rows:union|=np.frombuffer(row['seen'],dtype=np.uint8)
        assert union.all(),'At least one actual training row would be omitted'
        assert len({row['batches'] for row in rows})==1,'Unequal DDP batch counts'
        exposure=sum(row['count'] for row in rows);assert n<=exposure<n+self.world
        steps=[int(v['step']) for v in self.trainer.optimizer.state.values() if 'step' in v]
        if self.initial_step:assert steps and min(steps)==max(steps)==self.initial_step
        else:assert not steps or max(steps)==0
        resume=os.environ.get('MERGED_RESUME')
        if resume:
            from peft import get_peft_model_state_dict
            from safetensors import safe_open
            base=self.trainer.model.module if hasattr(self.trainer.model,'module') else self.trainer.model
            adapters=get_peft_model_state_dict(base)
            with safe_open(str(Path(resume)/'adapter_model.safetensors'),framework='pt',device='cpu') as checkpoint:
                assert set(adapters)==set(checkpoint.keys())
                for name,value in adapters.items():
                    size=min(2,value.shape[0])
                    assert torch.equal(value[:size].detach().float().cpu(),checkpoint.get_slice(name)[:size].float())
        self.info={'world_size':self.world,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                   'quantized_base':quantized,
                   'retained_activation_layers':retained_layers,
                   'trainable_parameter_dtypes':sorted({str(p.dtype) for p in self.trainer.model.parameters() if p.requires_grad}),
                   'micro_batch':args.per_device_train_batch_size,'gradient_accumulation_steps':args.gradient_accumulation_steps,
                   'effective_batch':effective,'initial_global_step':self.initial_step,'max_steps':state.max_steps,
                   'train_rows':n,'validation_rows':len(self.trainer.eval_dataset),
                   'sampling':'length grouping, full corpus, equal-rank padding','all_training_rows_covered':True,
                   'padding_repeat_rows':exposure-n,'sampler_batches_per_rank':batches,
                   'length_group_reference_batch_size':sampler.reference_batch_size,
                   'optimizer_step_min':min(steps) if steps else 0,'optimizer_step_max':max(steps) if steps else 0,
                   'checkpoint_adapter_samples_matched':True if resume else None,
                   'initial_learning_rates':[g['lr'] for g in self.trainer.optimizer.param_groups]}
        torch.cuda.reset_peak_memory_stats()
        if self.rank==0:(self.run/'actual_runtime.json').write_text(json.dumps(self.info,indent=2)+'\n')

    def on_pre_optimizer_step(self,args,state,control,**kwargs):
        if state.global_step==self.initial_step:
            values=[signature(self.trainer.model,True)]
            if self.world>1:
                values=[None]*self.world;dist.all_gather_object(values,signature(self.trainer.model,True))
            assert len(set(values))==1,'DDP gradient samples differ'
            if self.rank==0:(self.run/'first_gradient_sync.json').write_text(json.dumps({'step':self.initial_step,'ranks':self.world,'verified':True,'scope':'DDP sample equality' if self.world>1 else 'finite local gradient sample'})+'\n')

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step>=state.max_steps:control.should_evaluate=True;control.should_save=True
        return control

    def on_log(self,args,state,control,logs=None,**kwargs):
        for name in ('loss','grad_norm','eval_loss'):
            if name in (logs or {}) and not math.isfinite(float(logs[name])):raise FloatingPointError(name)
        if self.rank==0:
            (self.run/'runtime_progress.json').write_text(json.dumps({'global_step':state.global_step,'max_steps':state.max_steps,
                'epoch':state.epoch,'logs':logs,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3},default=str,indent=2)+'\n')

    def on_save(self,args,state,control,**kwargs):
        minimum_free=float(os.environ.get('MINIMUM_FREE_GIB','25'))
        low=torch.tensor(int(shutil.disk_usage(args.output_dir).free<minimum_free*1024**3),device=args.device)
        if self.world>1:dist.all_reduce(low,op=dist.ReduceOp.MAX)
        if low.item():raise RuntimeError('Checkpoint saved; insufficient disk space to continue')

    def on_train_end(self,args,state,control,**kwargs):
        final=signature(self.trainer.model);assert final!=self.initial_hash
        row={'rank':self.rank,'parameter_sample_sha256':final,
             'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
             'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3}
        ranks=[row]
        if self.world>1:
            ranks=[None]*self.world;dist.all_gather_object(ranks,row)
        assert len({r['parameter_sample_sha256'] for r in ranks})==1
        assert state.global_step==state.max_steps and 1<=state.epoch<1.0001
        if self.rank==0:
            result=self.info|{'global_step':state.global_step,'epoch':state.epoch,'probe':False,
                             'sampled_gradient_and_weight_sync':'verified' if self.world>1 else 'single_gpu','ranks':ranks}
            (self.run/'training/audit.json').write_text(json.dumps(result,indent=2)+'\n')


callbacks_map['merged_audit']=MergedAudit
