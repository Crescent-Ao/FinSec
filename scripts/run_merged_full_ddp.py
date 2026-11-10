"""Durable three-GPU full-corpus job, then FinEval, MIA and Agent diagnostics."""
import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_full_finance import digest,write_json
from scripts.run_full_finance import latest_checkpoint,storage_check

PYTHON=ROOT/'.venv-eval/bin/python';FAST=ROOT/'.venv-train-fast/bin/python'


def now():return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve()
    lock=(run/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=run/'status.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'created_utc':now(),'stages':{}}
    state.update(driver_pid=os.getpid(),run=str(run),gpus=[0,1,2]);state.pop('error',None)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    logs=run/'logs';logs.mkdir(exist_ok=True)
    def update(phase):state.update(phase=phase,updated_utc=now());write_json(state_path,state)
    def stage(name,command,extra=None):
        if state['stages'].get(name,{}).get('status')=='passed':return
        update(name);entry={'started_utc':now(),'status':'running','command':list(map(str,command))};state['stages'][name]=entry
        with (logs/(name+'.log')).open('a') as log:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT)
            entry['pid']=child.pid;write_json(state_path,state)
            while child.poll() is None:time.sleep(10);update(name)
        entry.update(ended_utc=now(),returncode=child.returncode,status='passed' if child.returncode==0 else 'failed');write_json(state_path,state)
        if child.returncode:raise RuntimeError(name+' failed; inspect '+str(logs/(name+'.log')))
    try:
        state['storage']=storage_check(run)
        update('waiting_for_preparation')
        deadline=time.monotonic()+3600
        while not (run/'manifest.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Full-corpus preparation did not finish')
            time.sleep(10);update('waiting_for_preparation')
        manifest=json.loads((run/'manifest.json').read_text());assert manifest['data_scale']=='full_filtered'
        signature=digest(run/'manifest.json')
        if state.get('manifest_sha256') not in (None,signature):raise RuntimeError('Frozen merged manifest changed')
        state['manifest_sha256']=signature
        for split,e in manifest['splits'].items():
            assert digest(run/(split+'.records.jsonl'))==e['source_sha256']
            if split!='privacy_nonmembers':assert digest(run/(split+'.jsonl'))==e['training_file_sha256']
        state['data']={s:{k:e[k] for k in ('rows','tokens','source_counts')} for s,e in manifest['splits'].items()}
        retirement=run/'old_study_retirement.json';update('waiting_for_previous_checkpoints')
        while not retirement.exists() or json.loads(retirement.read_text()).get('status')!='completed':
            if time.monotonic()>deadline:raise TimeoutError('Old study retirement did not finish')
            time.sleep(10);update('waiting_for_previous_checkpoints')
        update('waiting_for_gpus')
        while True:
            used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if max(map(int,used.split()))<1000:break
            if time.monotonic()>deadline:raise TimeoutError('Three GPUs did not become idle')
            time.sleep(10)
        baseline=run/'baseline/M0/fineval'
        if not baseline.exists():
            old=Path(manifest['old_manifest']).parent;shutil.copytree(old/'baseline/M0/fineval',baseline)
        base=json.loads((baseline/'summary.json').read_text());assert base['examples']==1151 and base['subjects']==34
        if not (run/'stress.jsonl').exists():
            long=(run/'longest_probe.jsonl').read_text().strip()
            with (run/'train.jsonl').open() as f:short=min([next(f).strip() for _ in range(64)],key=len)
            (run/'stress.jsonl').write_text((long+'\n')*72)
            (run/'stress_mixed.jsonl').write_text(''.join((short if i%4==0 else long)+'\n' for i in range(72)))
        for name,mixed in [('ddp3_4k_uniform',False),('ddp3_4k_mixed',True)]:
            directory=run/name
            if not (directory/'metrics.json').exists():
                command=[PYTHON,ROOT/'scripts/benchmark_training.py',run,'--name',name,'--gpus','0,1,2',
                         '--micro','4','--global-batch','24','--stress','--liger']
                if mixed:command+=['--stress-file',run/'stress_mixed.jsonl']
                stage(name,command)
            metrics=json.loads((directory/'metrics.json').read_text())
            assert metrics['gradient_and_weight_sync']=='verified' and metrics['effective_batch']==24
            assert max(r['peak_reserved_GiB'] for r in metrics['ranks'])<=22.0,'Insufficient 4K memory headroom'
        checkpoint=latest_checkpoint(run)
        config={'model':str(ROOT/'models/Qwen3.5-9B'),'model_type':'qwen3_5','use_hf':True,'tuner_type':'lora',
                'quant_method':'bnb','quant_bits':4,'bnb_4bit_compute_dtype':'bfloat16','bnb_4bit_quant_type':'nf4',
                'bnb_4bit_use_double_quant':True,'torch_dtype':'bfloat16','dataset':[str(run/'train.jsonl')],
                'val_dataset':[str(run/'validation.jsonl')],'split_dataset_ratio':0,'max_length':4096,'truncation_strategy':'delete',
                'per_device_train_batch_size':4,'per_device_eval_batch_size':1,'gradient_accumulation_steps':2,
                'num_train_epochs':1,'learning_rate':2e-5,'warmup_ratio':.05,'lr_scheduler_type':'cosine',
                'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,'target_modules':['all-linear'],
                'freeze_vit':True,'freeze_aligner':True,'gradient_checkpointing':True,'attn_impl':'sdpa',
                'add_non_thinking_prefix':True,'use_logits_to_keep':True,'use_liger_kernel':True,
                'liger_kernel_config':{'rope':False,'cross_entropy':False,'fused_linear_cross_entropy':True,'rms_norm':False,'swiglu':False},
                'seed':42,'data_seed':42,'group_by_length':True,'packing':False,
                'dataset_num_proc':8,'dataloader_num_workers':0,'eval_strategy':'no','save_strategy':'steps','save_steps':250,
                'load_best_model_at_end':False,'save_total_limit':2,'logging_steps':1,'report_to':'none',
                'ddp_find_unused_parameters':False,'auto_find_batch_size':False,'load_args':False,'ignore_data_skip':False,
                'external_plugins':[str(ROOT/'scripts/merged_training_audit.py')],'callbacks':['merged_audit'],
                'output_dir':str(run/'training')}
        if checkpoint:config['resume_from_checkpoint']=str(checkpoint)
        path=run/'sft_config.json';write_json(path,config)
        state['planned_steps']=math.ceil(manifest['splits']['train']['rows']/24)
        train_env={'CUDA_VISIBLE_DEVICES':'0,1,2','MERGED_RUN':str(run),'USE_HUB_KERNELS':'NO',
                   'HF_DATASETS_CACHE':str(run/'dataset_cache'),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
                   'NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}
        if checkpoint:train_env['MERGED_RESUME']=str(checkpoint)
        if not (run/'training/audit.json').exists():
            stage('training',[FAST,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3',
                              ROOT/'scripts/swift_config_entry.py',path],train_env)
        audit=json.loads((run/'training/audit.json').read_text());assert 1<=audit['epoch']<1.0001 and audit['all_training_rows_covered']
        stage('fineval',[PYTHON,ROOT/'scripts/complete_distilled_finance.py',run,'--baseline',run/'baseline','--result-prefix','merged_full_ddp3','--gpu','0'],{'CUDA_VISIBLE_DEVICES':'0'})
        stage('privacy',[PYTHON,'-m','fineval_pipeline','--config',run/'evaluation.yaml','run','--models','M0','M1','--tasks','privacy'],{'CUDA_VISIBLE_DEVICES':'0'})
        stage('agent_diagnostic',[PYTHON,'-m','fineval_pipeline','--config',run/'evaluation.yaml','run','--models','M1','--tasks','agent'],{'CUDA_VISIBLE_DEVICES':'0'})
        update('completed')
    except BaseException as error:
        state['error']=str(error);update('failed');raise


if __name__=='__main__':main()
