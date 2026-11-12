#!/usr/bin/env python3
"""Source-release entry point for FinSec training, evaluation and crypto auditing."""
from pathlib import Path
import argparse,json,os,subprocess,sys,importlib.util,platform,math
ROOT=Path(__file__).resolve().parent

def doctor():
    info={'platform':platform.platform(),'python':sys.version.split()[0],
          'python_3_11':sys.version_info[:2]==(3,11),
          'finsec_adapter':(ROOT/'models/FinSec/adapter_model.safetensors').is_file(),
          'finance_adapter':(ROOT/'models/finance_sft/adapter_model.safetensors').is_file(),
          'offline_demo':(ROOT/'demo/index.html').exists(),
          'packages':{x:importlib.util.find_spec(x) is not None for x in ['torch','transformers','peft','bitsandbytes','swift']}}
    import shutil
    info['node']=shutil.which('node');info['bubblewrap']=shutil.which('bwrap')
    info['gpu_jobs_started']=0
    print(json.dumps(info,ensure_ascii=False,indent=2))

def train_plan(a):
    profiles={'finance_demo':(2e-5,4096,24),'safety_tools':(5e-6,4096,25),'safety_long':(2e-6,16384,8),'dpo':(2e-6,4096,16),'finsec':(1e-6,4096,48)}
    lr,length,global_batch=profiles[a.stage];world=len(a.gpus.split(','));micro=a.micro_batch
    if global_batch%(world*micro):raise ValueError(f'Effective batch {global_batch} must be divisible by micro batch × GPU count')
    parent=a.parent
    if not parent and a.stage=='safety_tools':parent=str(ROOT/'models/finance_sft')
    if a.stage not in ['finance_demo','safety_tools'] and not parent:raise ValueError('--parent is required: use the correct preceding trained adapter')
    files={'finance_demo':('datasets/finance_examples/train.sample.jsonl',None),
      'safety_tools':('datasets/stage2/sft_train.jsonl','datasets/stage2/sft_validation.jsonl'),
      'safety_long':('datasets/stage2/sft_long_train.jsonl','datasets/stage2/sft_long_validation.jsonl'),
      'dpo':('datasets/stage2/dpo_train.jsonl','datasets/stage2/dpo_validation.jsonl'),
      'finsec':('datasets/finsec/sft_train.jsonl','datasets/finsec/sft_validation.jsonl')}
    train,val=files[a.stage];output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    opts={'model':str(Path(a.model_dir).resolve()),'model_type':'qwen3_5','use_hf':True,'tuner_type':'lora',
      'load_args':False,'torch_dtype':'bfloat16','quant_method':'bnb','quant_bits':4,'bnb_4bit_quant_type':'nf4',
      'bnb_4bit_compute_dtype':'bfloat16','bnb_4bit_use_double_quant':True,'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,
      'target_modules':['all-linear'],'freeze_vit':True,'freeze_aligner':True,'gradient_checkpointing':True,
      'dataset':[str(Path(a.data).resolve()) if a.data else str(ROOT/train)],'val_dataset':[str(ROOT/val)] if val else [],
      'split_dataset_ratio':0,'num_train_epochs':1,'max_length':length,'per_device_train_batch_size':micro,
      'per_device_eval_batch_size':1,'gradient_accumulation_steps':global_batch//(world*micro),
      'learning_rate':lr,'warmup_ratio':.05,'lr_scheduler_type':'cosine','optim':'adamw_torch_fused',
      'attn_impl':'sdpa','add_non_thinking_prefix':True,'enable_thinking':False,
      'dataset_num_proc':2,'dataloader_num_workers':0,'eval_strategy':'epoch' if val else 'no',
      'save_strategy':'steps','save_steps':50,'save_total_limit':2,'logging_steps':1,'report_to':['tensorboard'],
      'output_dir':str(output/'checkpoints'),'seed':42,'data_seed':42,'ddp_find_unused_parameters':False}
    if parent:opts['adapters']=[str(Path(parent).resolve())]
    algorithm='dpo' if a.stage=='dpo' else 'sft'
    if algorithm=='dpo':opts.update(rlhf_type='dpo',ref_adapters=[str(Path(parent).resolve())],beta=.1,rpo_alpha=.2)
    else:opts['group_by_length']=True
    if a.max_steps:opts['max_steps']=a.max_steps
    plan={'algorithm':algorithm,'stage':a.stage,'arguments':opts,'effective_batch':global_batch,
      'scope':'user-provided training data' if a.data else 'local dataset path (not distributed with source)',
      'note':'Portable batch setting; datasets and adapters are supplied separately.'}
    if a.validation_data:
        opts['val_dataset']=[str(Path(a.validation_data).resolve())];opts['eval_strategy']='epoch'
    cfg=output/'train_config.json';cfg.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    command=[a.python,str(ROOT/'scripts/stage2_swift_entry.py'),str(cfg)]
    if world>1:command=[a.python,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(ROOT/'scripts/stage2_swift_entry.py'),str(cfg)]
    print(json.dumps({'config':str(cfg),'stage':a.stage,'scope':plan['scope'],'global_batch':global_batch,'command':command,'execution_requested':a.execute},ensure_ascii=False,indent=2))
    if a.execute:
        for p in [opts['model'],*opts['dataset'],*opts.get('adapters',[])]:
            if not Path(p).exists():raise FileNotFoundError(p)
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus),check=True)

def setup_harmbench():
    meta=json.loads((ROOT/'configs/harmbench.lock.json').read_text());target=ROOT/'third_party/HarmBench'
    if target.exists():
        actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if actual!=meta['revision']:raise ValueError('Existing HarmBench revision differs')
    else:
        subprocess.run(['git','init',str(target)],check=True,stdout=subprocess.DEVNULL)
        bundle=ROOT/'third_party/HarmBench.bundle'
        if bundle.exists():
            subprocess.run(['git','-C',str(target),'bundle','unbundle',str(bundle)],check=True)
            boundaries=meta.get('shallow_boundaries',[])
            if boundaries:(target/'.git/shallow').write_text('\n'.join(boundaries)+'\n')
        else:
            subprocess.run(['git','-C',str(target),'remote','add','origin',meta['upstream']],check=True)
            subprocess.run(['git','-C',str(target),'fetch','--depth','1','origin',meta['revision']],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',meta['revision']],check=True)
    print('HarmBench official source ready:',target)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    verify=sub.add_parser('verify');verify.add_argument('--bundle-root',type=Path)
    sub.add_parser('doctor');sub.add_parser('setup-harmbench');sub.add_parser('download-base');sub.add_parser('fetch-fineval')
    ev=sub.add_parser('eval-fineval');ev.add_argument('--model-dir',required=True);ev.add_argument('--variant',choices=['base','finance','safety_tools','dpo','FinSec'],default='FinSec')
    ev.add_argument('--adapter',help='Override the adapter path for any non-base stage')
    ev.add_argument('--output',default='runs/fineval');ev.add_argument('--limit-per-subject',type=int,default=0);ev.add_argument('--gpus',default='0')
    suite=sub.add_parser('eval-suite',help='Plan, execute or collect a five-stage comparison')
    suite.add_argument('--config',default='configs/evaluation_suite.example.json')
    suite.add_argument('--tasks',nargs='+',choices=['fineval','privacy','agent','gcg','pair'])
    mode=suite.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--collect',action='store_true')
    demo=sub.add_parser('demo');demo.add_argument('--port',type=int,default=8765)
    t=sub.add_parser('train-plan');t.add_argument('--stage',choices=['finance_demo','safety_tools','safety_long','dpo','finsec'],required=True)
    t.add_argument('--model-dir',required=True);t.add_argument('--parent');t.add_argument('--data');t.add_argument('--validation-data');t.add_argument('--output',default='runs/train_plan')
    t.add_argument('--micro-batch',type=int,default=1);t.add_argument('--gpus',default='0');t.add_argument('--python',default=sys.executable)
    t.add_argument('--max-steps',type=int);t.add_argument('--execute',action='store_true')
    a=sub.add_parser('audit');a.add_argument('--file',required=True);a.add_argument('--model-dir',required=True)
    a.add_argument('--adapter',default=str(ROOT/'models/FinSec'));a.add_argument('--python',default=sys.executable)
    a.add_argument('--node',default='node');a.add_argument('--output',default='runs/live_agent');a.add_argument('--gpus',default='0')
    a.add_argument('--max-turns',type=int,default=12)
    args=p.parse_args()
    if args.action=='eval-fineval' and args.variant=='base' and args.adapter:
        p.error('The base variant must not load an adapter')
    if args.action=='verify':
        from verify import verify_release
        verify_release(args.bundle_root)
    elif args.action=='doctor':doctor()
    elif args.action=='eval-suite':
        from scripts.experiment_suite import run
        run(args)
    elif args.action=='setup-harmbench':setup_harmbench()
    elif args.action=='download-base':subprocess.run([sys.executable,str(ROOT/'scripts/download_model.py')],cwd=ROOT,check=True)
    elif args.action=='fetch-fineval':
        from scripts.fetch_fineval import main as fetch
        fetch()
    elif args.action=='demo':
        from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
        from functools import partial
        print(f'Open http://127.0.0.1:{args.port}/demo/index.html',flush=True)
        ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(ROOT))).serve_forever()
    elif args.action=='train-plan':train_plan(args)
    elif args.action=='eval-fineval':
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
        from fineval_pipeline.core import Backend,model_identity
        from fineval_pipeline.finance import evaluate
        spec={'path':str(Path(args.model_dir).resolve()),'precision':'nf4','system_prompt':''}
        if args.variant!='base':
            folder={'finance':'finance_sft','safety_tools':'safety_tools','dpo':'dpo','FinSec':'FinSec'}[args.variant]
            spec['adapter']=str(Path(args.adapter).resolve()) if args.adapter else str(ROOT/'models'/folder)
        import torch
        torch.manual_seed(42)
        identity=model_identity(spec);backend=Backend(spec)
        cfg={'fineval':{'data_dir':str(ROOT/'evaluation_data/FinEval/academic/val'),'source_manifest':str(ROOT/'configs/fineval_reference.lock.json'),
            'split':'val','limit_per_subject':args.limit_per_subject}}
        result=evaluate(cfg,args.variant,backend,identity,Path(args.output).resolve())
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.action=='audit':
        command=[args.node,str(ROOT/'submission_tools/live_agent.mjs'),'--file',str(Path(args.file).resolve()),'--model-dir',str(Path(args.model_dir).resolve()),
                 '--adapter',str(Path(args.adapter).resolve()) if args.adapter else '','--python',args.python,'--output',str(Path(args.output).resolve()),'--max-turns',str(args.max_turns)]
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus),check=True)

if __name__=='__main__':main()
