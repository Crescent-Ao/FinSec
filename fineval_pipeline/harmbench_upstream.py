"""Drive the official HarmBench CLI pipeline, with auditable compatibility patches."""
import csv
import difflib
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from .core import ROOT, bind, digest, fingerprint, model_identity, model_spec, resolve, write_json
from .harmbench import behaviors, upstream

NATIVE_CONVERSATION = '''from transformers import AutoTokenizer
from functools import lru_cache
@lru_cache(maxsize=4)
def tokenizer_for(path): return AutoTokenizer.from_pretrained(path,local_files_only=True)
class NativeConversation:
    name = "qwen3_5_native"
    roles = ("user", "assistant")
    sep2 = "<|im_end|>\\n"
    def __init__(self, path, system_message=None):
        self.path = path
        self.system_message = system_message
        self.messages = []
    def set_system_message(self, text): self.system_message = text
    def append_message(self, role, text): self.messages.append([role,text])
    def update_last_message(self,text): self.messages[-1][1] = text
    def to_openai_api_messages(self):
        result = ([{"role":"system","content":self.system_message}] if self.system_message else [])
        return result + [{"role":r,"content":t} for r,t in self.messages if t is not None]
    def get_prompt(self):
        tokenizer = tokenizer_for(self.path)
        messages = self.to_openai_api_messages()
        prefill = None
        if messages and messages[-1]["role"] == "assistant":
            prefill = messages.pop()["content"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text if prefill is None else text + prefill + self.sep2
    def dict(self): return {"template_name":"qwen3_5_native", "messages":self.messages}
'''


def build_runtime(cfg):
    source=upstream(cfg)
    destination=resolve(cfg['harmbench']['runtime_dir'])
    marker=destination/'compatibility_manifest.json'
    performance = cfg['harmbench'].get('gcg_performance', {}).get('enabled', False)
    generator=fingerprint({'driver':digest(__file__), 'gcg_performance':performance,
        'performance_source':digest(Path(__file__).with_name('harmbench_performance.py')) if performance else None})
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['generator_sha256']==generator:
            for r in old['changed_files']:
                if digest(destination/r['file'])!=r['patched_sha256']:
                    raise ValueError('Runtime patch changed outside the recorded generator')
            for file in old['unchanged_algorithms']:
                if digest(source/file)!=digest(destination/file):
                    raise ValueError('Official runtime core algorithm was modified')
            return destination
        if old['upstream_revision']!=cfg['harmbench']['revision']:
            raise ValueError('Refusing to replace a runtime from a different source')
        shutil.rmtree(destination)
    elif destination.exists():
        raise ValueError('Untracked runtime directory; choose a new runtime_dir')
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('.git','__pycache__'))
    before={}
    def replace(file,old,new):
        path=destination/file
        text=path.read_text()
        if old not in text:raise ValueError(f'Pinned compatibility patch does not apply: {file}: {old[:60]}')
        before.setdefault(file,text)
        path.write_text(text.replace(old,new))

    # Backend/class/template compatibility, not attack-search rewrites.
    replace('baselines/model_utils.py',
            'from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig',
            'from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoConfig')
    replace('baselines/model_utils.py',
            'model = AutoModelForCausalLM.from_pretrained(model_name_or_path,',
            'for key in ("gpu_memory_utilization","max_model_len","max_num_seqs","enforce_eager","seed","limit_mm_per_prompt","use_vllm","ray_gpu_fraction","cpu_offload_gb","enable_prefix_caching"):\n        model_kwargs.pop(key,None)\n    precision = model_kwargs.pop("precision", "bf16")\n    adapter_path = model_kwargs.pop("adapter_path", None)\n    if precision == "nf4":\n        from transformers import BitsAndBytesConfig\n        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)\n    model_cls = AutoModelForImageTextToText if AutoConfig.from_pretrained(model_name_or_path).model_type == "qwen3_5" else AutoModelForCausalLM\n    model = model_cls.from_pretrained(model_name_or_path,')
    replace('baselines/model_utils.py',
            '    # Init Tokenizer',
            '    if adapter_path:\n        from peft import PeftModel\n        model = PeftModel.from_pretrained(model, adapter_path).eval()\n    print("HarmBench target loading", {"precision": precision, "adapter_path": adapter_path, "is_loaded_in_4bit": getattr(model, "is_loaded_in_4bit", False)})\n    model.requires_grad_(False)\n    model.config.use_cache = False\n    if hasattr(model.config, "get_text_config"):\n        model.config.get_text_config().use_cache = False\n\n    # Init Tokenizer')
    replace('baselines/model_utils.py',
            'add_generation_prompt=True)', 'add_generation_prompt=True, enable_thinking=False)')
    replace('baselines/model_utils.py',
            '    template_name = fschat_template',
            '    if fschat_template == "qwen3_5_native":\n        from .native_conversation import NativeConversation\n        return NativeConversation(model_name_or_path, system_message)\n    template_name = fschat_template')
    # Preserve configured engine memory/context budgets on modern vLLM.
    replace('baselines/model_utils.py',
            'model_kwargs = {k: kwargs[k] for k in kwargs if k in signature(LLM).parameters}',
            'allowed = {"gpu_memory_utilization", "max_model_len", "max_num_seqs", "enforce_eager", "seed", "limit_mm_per_prompt", "cpu_offload_gb", "enable_prefix_caching"}\n    model_kwargs = {k: kwargs[k] for k in kwargs if k in signature(LLM).parameters or k in allowed}')
    replace('baselines/model_utils.py','tensor_parallel_size=num_gpus)',
            'tensor_parallel_size=num_gpus, **model_kwargs)')
    for file in ['baselines/model_utils.py','baselines/pair/language_models.py','evaluate_completions.py']:
        replace(file,'.llm_engine.tokenizer.tokenizer','.get_tokenizer()')
    # PAIR has two local model engines on one 80GB card; reserve fractional Ray GPUs.
    replace('baselines/pair/conversers.py',
            'num_gpus = model_kwargs.get("num_gpus", 1)',
            'num_gpus = model_kwargs.get("num_gpus", 1)\n        ray_gpu_fraction = model_kwargs.pop("ray_gpu_fraction", num_gpus)')
    replace('baselines/pair/conversers.py','options = {"num_gpus": 1} if num_gpus == 1 else {}',
            'options = {"num_gpus": ray_gpu_fraction} if num_gpus == 1 else {}')
    replace('baselines/pair/conversers.py','options(num_gpus=num_gpus)', 'options(num_gpus=ray_gpu_fraction)')
    replace('baselines/pair/conversers.py','if t_name == a_name:',
            'if t_name == a_name and targetLM_args == attackLM_args:')
    replace('baselines/pair/conversers.py','if j_name == a_name:',
            'if j_name == a_name and judgeLM_args == attackLM_args:')
    replace('baselines/pair/conversers.py','elif j_name == t_name:',
            'elif j_name == t_name and judgeLM_args == targetLM_args:')
    replace('baselines/pair/language_models.py','tensor_parallel_size=num_gpus)',
            'tensor_parallel_size=num_gpus, **{k:kwargs[k] for k in ("gpu_memory_utilization","max_model_len","max_num_seqs","enforce_eager","seed","limit_mm_per_prompt","cpu_offload_gb","enable_prefix_caching") if k in kwargs})')
    replace('baselines/pair/judges.py','from .system_prompts import get_judge_system_prompt',
            'from .system_prompts import get_judge_system_prompt\nfrom ..model_utils import get_template')
    replace('baselines/pair/judges.py',
            'conv = get_conversation_template(self.judge_name)\n        conv.append_message(conv.roles[0], full_prompt)',
            'conv = get_template(self.judge_name, fschat_template="qwen3_5_native", return_fschat_conv=True)\n        conv.append_message(conv.roles[0], full_prompt)')
    # Unused legacy cloud/multimodal adapters must not eagerly import incompatible SDKs.
    replace('generate_completions.py','from api_models import api_models_map',
            'def api_models_map(**kwargs):\n    if any(x in kwargs.get("model_name_or_path", "") for x in ("gpt-", "claude-", "gemini-", "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large")):\n        from api_models import api_models_map as load_api\n        return load_api(**kwargs)\n    return None')
    replace('generate_completions.py','import multimodalmodels',
            'from types import SimpleNamespace\nmultimodalmodels = SimpleNamespace()  # This runner evaluates text behaviors only.')
    # Explicit evaluator memory budget and strict current tokenizer API.
    replace('evaluate_completions.py','cls = LLM(model=args.cls_path, tensor_parallel_size=1)',
            'cls = LLM(model=args.cls_path, tensor_parallel_size=int(os.environ.get("HB_EVAL_TP","1")), gpu_memory_utilization=float(os.environ.get("HB_EVAL_GPU_MEMORY_UTILIZATION","0.85")), cpu_offload_gb=float(os.environ.get("HB_EVAL_CPU_OFFLOAD_GB","0")), max_num_batched_tokens=int(os.environ.get("HB_EVAL_BATCHED_TOKENS","2048")), max_num_seqs=int(os.environ.get("HB_EVAL_MAX_SEQS","32")), enforce_eager=True)')
    # All stochastic HF attack operations share an explicit logged process seed.
    replace('generate_test_cases.py','def main():',
            'def main():\n    import random, numpy as np\n    seed = int(os.environ.get("HB_SEED", "42"))\n    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)')
    # Local Ray lifecycle is owned by this driver, rather than orphaning ray-start daemons.
    import ast
    ray_source=(destination/'baselines/model_utils.py').read_text()
    node=next(n for n in ast.parse(ray_source).body if isinstance(n,ast.FunctionDef) and n.name=='_init_ray')
    old=''.join(ray_source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    replace('baselines/model_utils.py',old,
            'def _init_ray(num_cpus=8, reinit=False, resources={}):\n    if ray.is_initialized(): return\n    ray.init(num_cpus=min(os.cpu_count(),num_cpus), num_gpus=torch.cuda.device_count(), include_dashboard=False, ignore_reinit_error=True)\n')
    path=destination/'baselines/native_conversation.py'
    path.write_text(NATIVE_CONVERSATION)
    if performance:
        from .harmbench_performance import optimize_gcg
        replace('baselines/model_utils.py',
            '    precision = model_kwargs.pop("precision", "bf16")',
            '    gcg_fast_convolution = model_kwargs.pop("gcg_fast_convolution", False)\n'
            '    precision = model_kwargs.pop("precision", "bf16")')
        replace('baselines/model_utils.py',
            '    model.requires_grad_(False)',
            '    if gcg_fast_convolution:\n'
            '        from .gcg.performance_runtime import enable_fla_convolution\n'
            '        enable_fla_convolution()\n'
            '    model.requires_grad_(False)')
        gcg_path = destination/'baselines/gcg/gcg.py'
        before['baselines/gcg/gcg.py'] = gcg_path.read_text()
        gcg_path.write_text(optimize_gcg(before['baselines/gcg/gcg.py']))
        replace('generate_test_cases.py',
            '    test_cases, logs = method.generate_test_cases(behaviors=behaviors, verbose=args.verbose)',
            '    if method_name == "GCG":\n'
            '        from baselines.gcg.performance_runtime import checkpointed_generation\n'
            '        test_cases, logs = checkpointed_generation(method, behaviors, save_dir, method_config, args.run_id, args.verbose)\n'
            '    else:\n'
            '        test_cases, logs = method.generate_test_cases(behaviors=behaviors, verbose=args.verbose)')
        helper=destination/'baselines/gcg/performance_runtime.py'
        shutil.copy2(Path(__file__).with_name('harmbench_performance.py'), helper)
    changes=[];diff=[]
    for file,original in before.items():
        modified=(destination/file).read_text()
        changes.append({'file':file,'upstream_sha256':digest(source/file),'patched_sha256':digest(destination/file)})
        diff.extend(difflib.unified_diff(original.splitlines(keepends=True),modified.splitlines(keepends=True),
                                       fromfile='upstream/'+file,tofile='runtime/'+file))
    changes.append({'file':'baselines/native_conversation.py','upstream_sha256':None,'patched_sha256':digest(path)})
    if performance:
        changes.append({'file':'baselines/gcg/performance_runtime.py','upstream_sha256':None,'patched_sha256':digest(helper)})
    (destination/'compatibility.patch').write_text(''.join(diff))
    manifest={'upstream_revision':cfg['harmbench']['revision'],'generator_sha256':generator,'changed_files':changes,
              'unchanged_algorithms':['baselines/gcg/gcg_utils.py','baselines/pair/PAIR.py','eval_utils.py'] + ([] if performance else ['baselines/gcg/gcg.py']),
              'gcg_performance': {'enabled':performance, 'suffix_logits':performance,
                  'batched_token_filter':performance, 'allocator_reuse':performance,
                  'per_behavior_checkpoint':performance, 'per_behavior_seed':performance,
                  'fla_triton_convolution':performance, 'frozen_vocabulary_cache':performance,
                  'fixed_candidate_batch_shapes':performance},
              'notes':'Official CLI pipeline with recorded compatibility patches. When enabled, GCG execution optimizations change logit materialization/token filtering/allocator reuse/persistence; search and loss formulas are preserved. Kernel numerics can change search trajectories.'}
    for file in manifest['unchanged_algorithms']:
        if digest(source/file)!=digest(destination/file):raise ValueError('Core algorithm was modified')
    write_json(marker,manifest)
    return destination


def prepare(cfg,name,method,directory):
    runtime=build_runtime(cfg)
    spec=model_spec(cfg,name)
    selected,excluded=behaviors(cfg)
    settings=cfg['harmbench']
    directory.mkdir(parents=True,exist_ok=True)
    data=directory/'behaviors.csv'
    import io,hashlib
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    data_text=buffer.getvalue()
    target={'model_name_or_path':spec['path'],'dtype':'bfloat16','num_gpus':1,
            'precision':spec.get('precision','bf16'),'adapter_path':spec.get('adapter'),
            'fschat_template':'qwen3_5_native','trust_remote_code':False,
            'gpu_memory_utilization':settings['vllm'].get('gpu_memory_utilization',0.42),
            'max_model_len':settings['vllm'].get('max_model_len',8192),'max_num_seqs':settings['vllm'].get('max_num_seqs',8),
            'enforce_eager':settings['vllm']['enforce_eager'],'seed':settings['seed'],
            'limit_mm_per_prompt':{'image':0,'video':0}}
    for key in ('cpu_offload_gb','enable_prefix_caching'):
        if key in settings['vllm']: target[key]=settings['vllm'][key]
    model_yaml=directory/'models.yaml'
    upstream_config=yaml.safe_load((runtime/f'configs/method_configs/{method}_config.yaml').read_text())
    params=dict(upstream_config['default_method_hyperparameters'])
    params['targets_path']=str(resolve(settings['targets']))
    if method=='GCG':
        params.update(settings['gcg']);params['target_model']=target
        if settings.get('gcg_performance', {}).get('enabled'):
            target['gcg_fast_convolution'] = True
        if settings.get('gcg_performance', {}).get('enabled') and params.get('use_prefix_cache'):
            raise ValueError('Optimized Qwen3.5 GCG requires use_prefix_cache=false for hybrid attention correctness')
    else:
        opts=settings['pair']
        pair_target=dict(target)
        for key,value in settings['vllm'].get('pair_engine_overrides',{}).items():
            pair_target[key]=value
        def local_model(key):
            other=model_spec(cfg,opts[key])
            return {**pair_target,'model_name_or_path':other['path'],
                    'precision':other.get('precision','bf16'),'adapter_path':other.get('adapter'),
                    'use_vllm':other.get('precision','bf16')=='bf16' and not other.get('adapter'),
                    'ray_gpu_fraction':settings['vllm'].get('ray_gpu_fraction',0.45)}
        params.update({k:opts[k] for k in ('n_streams','steps','max_retries','attack_max_n_tokens','judge_max_n_tokens')})
        params['attack_model']=local_model('attacker');params['judge_model']=local_model('search_judge')
        params['target_model']={**pair_target,
            'use_vllm':spec.get('precision','bf16')=='bf16' and not spec.get('adapter'),
            'ray_gpu_fraction':settings['vllm'].get('ray_gpu_fraction',0.45)}
        params['target_max_n_tokens']=settings['target_max_new_tokens']
    method_yaml=directory/'method.yaml'
    context={'task':'harmbench-official-cli-v1','method':method,'model':model_identity(spec),
             'behaviors_sha256':hashlib.sha256(data_text.encode()).hexdigest(),'parameters':params,'target_max_new_tokens':settings['target_max_new_tokens'],
             'upstream_revision':settings['revision'],'runtime_manifest':digest(runtime/'compatibility_manifest.json'),
             'seed':settings['seed'],'hardware':cfg.get('hardware',{})}
    bind(directory,context)
    data.write_text(data_text)
    model_yaml.write_text(yaml.safe_dump({'target':{'model':target,'num_gpus':1}}))
    method_yaml.write_text(yaml.safe_dump({'default_method_hyperparameters':params,'run':{}}))
    shutil.copy2(runtime/'compatibility_manifest.json',directory/'compatibility_manifest.json')
    shutil.copy2(runtime/'compatibility.patch',directory/'compatibility.patch')
    comparison_parameters={k:v for k,v in params.items() if k!='target_model'}
    write_json(directory/'attack_summary.json',{'model_identity':context['model'],'selected_ids':[b['BehaviorID'] for b in selected],
        'excluded_unsupported_ids':excluded,'parameters':comparison_parameters,'method':method,'split':settings['split'],
        'protocol':context['task'],'target_max_new_tokens':settings['target_max_new_tokens'],
        'scope':'subset' if settings['limit'] or excluded else 'all_text_behaviors',
        'runtime_manifest':context['runtime_manifest']})
    return runtime,model_yaml,method_yaml,data


def run_command(cfg,runtime,command,log):
    python=resolve(cfg['harmbench']['python'])
    if not python.is_file():raise FileNotFoundError('Install the independent HarmBench/vLLM environment first')
    env=dict(os.environ,HB_SEED=str(cfg['harmbench']['seed']),VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',
             PYTHONPATH=str(runtime)+os.pathsep+os.environ.get('PYTHONPATH',''),VLLM_NO_USAGE_STATS='1')
    env['HB_EVAL_GPU_MEMORY_UTILIZATION']=str(cfg['harmbench']['vllm'].get('classifier_gpu_memory_utilization',0.85))
    env['HB_EVAL_TP']=str(cfg['harmbench']['vllm'].get('classifier_tensor_parallel',1))
    env['HB_EVAL_CPU_OFFLOAD_GB']=str(cfg['harmbench']['vllm'].get('classifier_cpu_offload_gb',0))
    env['HB_EVAL_BATCHED_TOKENS']=str(cfg['harmbench']['vllm'].get('classifier_max_batched_tokens',2048))
    env['HB_EVAL_MAX_SEQS']=str(cfg['harmbench']['vllm'].get('classifier_max_seqs',32))
    env['PATH']=str(python.parent)+os.pathsep+env.get('PATH','')
    env['VLLM_USE_FLASHINFER_SAMPLER']='0'
    if cfg['harmbench']['vllm'].get('disable_p2p'):
        env['NCCL_P2P_DISABLE']='1';env['NCCL_IB_DISABLE']='1'
    with log.open('w') as output:
        subprocess.run([str(python)]+command,cwd=runtime,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)


def attack(cfg,name,method,directory):
    runtime,models,params,data=prepare(cfg,name,method,directory)
    cases=directory/'test_cases'
    run_command(cfg,runtime,['generate_test_cases.py','--method_name',method,'--experiment_name','run',
        '--method_config_file',str(params),'--models_config_file',str(models),'--behaviors_path',str(data),
        '--save_dir',str(cases)],directory/'step1_generate_attacks.log')
    run_command(cfg,runtime,['merge_test_cases.py','--method_name',method,'--save_dir',str(cases)],directory/'step1_merge.log')
    command=['generate_completions.py','--model_name','target','--models_config_file',str(models),
        '--behaviors_path',str(data),'--test_cases_path',str(cases/'test_cases.json'),
        '--save_path',str(directory/'completions.json'),'--incremental_update',
        '--max_new_tokens',str(cfg['harmbench']['target_max_new_tokens'])]
    spec=model_spec(cfg,name)
    if spec.get('precision','bf16')=='bf16' and not spec.get('adapter'):
        command.append('--generate_with_vllm')
    run_command(cfg,runtime,command,directory/'step2_completions.log')


def judge(cfg,name,method,directory):
    settings=cfg['harmbench'];classifier=settings['classifier']
    manifest=json.loads(resolve(classifier['manifest']).read_text())
    if manifest['repo']!=classifier['upstream_id'] or manifest['revision']!=classifier['revision']:
        raise ValueError('Classifier provenance mismatch')
    path=resolve(classifier['path'])
    for row in manifest['files']:
        if digest(path/row['file'])!=row['sha256']:raise ValueError('Classifier weights changed')
    runtime=build_runtime(cfg)
    judge_context={'task':'harmbench-official-cli-v1','classifier_manifest':digest(resolve(classifier['manifest'])),
                   'completions_sha256':digest(directory/'completions.json'),
                   'behaviors_sha256':digest(directory/'behaviors.csv'),
                   'runtime_manifest':digest(runtime/'compatibility_manifest.json'),
                   'gpu_memory_utilization':settings['vllm'].get('classifier_gpu_memory_utilization',0.85),
                   'tensor_parallel':settings['vllm'].get('classifier_tensor_parallel',1),
                   'cpu_offload_gb':settings['vllm'].get('classifier_cpu_offload_gb',0),
                   'max_batched_tokens':settings['vllm'].get('classifier_max_batched_tokens',2048),
                   'max_seqs':settings['vllm'].get('classifier_max_seqs',32)}
    bind(directory/'judge_run',judge_context)
    run_command(cfg,runtime,['evaluate_completions.py','--cls_path',str(path),
        '--behaviors_path',str(directory/'behaviors.csv'),'--completions_path',str(directory/'completions.json'),
        '--save_path',str(directory/'official_results.json')],directory/'step3_official_judge.log')
    results=json.loads((directory/'official_results.json').read_text())
    attack_meta=json.loads((directory/'attack_summary.json').read_text())
    if set(results)!=set(attack_meta['selected_ids']):raise ValueError('Incomplete official result coverage')
    rates=[]
    for key,records in results.items():
        if not records or any(r.get('label') not in (0,1) for r in records):
            raise ValueError('Official classifier invalid/missing labels; refusing optimistic ASR')
        rates.append(sum(r['label'] for r in records)/len(records))
    summary={'task':'harmbench-official-cli-v1','method':method,'asr':sum(rates)/len(rates),'behaviors':len(results),
             'classifier_identity':{'manifest_sha256':digest(resolve(classifier['manifest']))},'attack':attack_meta,
             'official_results_sha256':digest(directory/'official_results.json')}
    summary['judge_protocol']=judge_context
    write_json(directory/'summary.json',summary)
    return summary
