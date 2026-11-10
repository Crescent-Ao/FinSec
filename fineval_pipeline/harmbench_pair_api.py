"""Official PAIR search with a separately audited DeepSeek transport runtime."""
import copy
import csv
import difflib
import hashlib
import io
import json
from pathlib import Path
import shutil

import yaml

from .core import ROOT,bind,digest,fingerprint,model_identity,model_spec,resolve,write_json
from .harmbench import behaviors
from .harmbench_upstream import build_runtime

ALIAS='gpt-deepseek-flash'


def build_api_runtime(cfg):
    canonical=build_runtime(cfg)
    destination=resolve(cfg['harmbench'].get('api_runtime_dir','runtime/HarmBench_deepseek_pair_budget_v1'))
    marker=destination/'api_compatibility_manifest.json'
    identity=fingerprint({'generator':digest(__file__),'canonical':digest(canonical/'compatibility_manifest.json'),
                          'transport':digest(Path(__file__).with_name('deepseek_api.py'))})
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['fingerprint']==identity:
            for row in record['changed_files']:
                if digest(destination/row['file'])!=row['patched_sha256']:raise ValueError('API runtime changed outside its manifest')
            for file in record['unchanged_algorithms']:
                if digest(destination/file)!=digest(canonical/file):raise ValueError('Official PAIR algorithm changed')
            return destination
        shutil.rmtree(destination)
    elif destination.exists():raise ValueError('Refusing to replace an untracked API runtime')
    shutil.copytree(canonical,destination,ignore=shutil.ignore_patterns('__pycache__'))
    path=destination/'baselines/pair/conversers.py';before=path.read_text();source=before
    old='    if \'gpt-\' in model_name_or_path:\n        lm = GPT(model_name_or_path, token=token)'
    new=f'''    if model_name_or_path == {ALIAS!r}:
        from fineval_pipeline.deepseek_api import DeepSeek
        return DeepSeek(variant=model_kwargs.get("api_variant", "unknown"), model="deepseek-flash")
    if 'gpt-' in model_name_or_path:
        lm = GPT(model_name_or_path, token=token)'''
    if source.count(old)!=1:raise ValueError('Pinned API attacker transport patch no longer applies')
    source=source.replace(old,new)
    old='    if "gpt" in model_name_or_path:\n        judge_model = GPT(model_name=model_name_or_path, token=token)'
    new=f'''    if model_name_or_path == {ALIAS!r}:
        from fineval_pipeline.deepseek_api import DeepSeek
        judge_model = preloaded_model or DeepSeek(variant=model_kwargs.get("api_variant", "unknown"), model="deepseek-flash")
        return GPTJudge("gpt-4", max_n_tokens, judge_model)
    if "gpt" in model_name_or_path:
        judge_model = GPT(model_name=model_name_or_path, token=token)'''
    if source.count(old)!=1:raise ValueError('Pinned API judge transport patch no longer applies')
    path.write_text(source.replace(old,new))
    record={'fingerprint':identity,'canonical_manifest_sha256':digest(canonical/'compatibility_manifest.json'),
        'transport_sha256':digest(Path(__file__).with_name('deepseek_api.py')),
        'changed_files':[{'file':'baselines/pair/conversers.py','upstream_sha256':digest(canonical/'baselines/pair/conversers.py'),
                          'patched_sha256':digest(path)}],
        'unchanged_algorithms':['baselines/pair/PAIR.py','baselines/pair/common.py','baselines/pair/system_prompts.py','eval_utils.py'],
        'note':'Official PAIR search/prompt/extraction/scoring unchanged; transport maps the GPT message branch to DeepSeek Flash with a shared budget ledger.'}
    write_json(marker,record)
    (destination/'api_compatibility.patch').write_text(''.join(difflib.unified_diff(before.splitlines(keepends=True),
        path.read_text().splitlines(keepends=True),fromfile='canonical/conversers.py',tofile='api_runtime/conversers.py')))
    return destination


def prepare(cfg,name,variant,directory,runtime):
    settings=cfg['harmbench'];opts=settings['pair'];spec=model_spec(cfg,name)
    if spec.get('precision')!='nf4':raise ValueError('PAIR target must preserve the NF4 experiment baseline')
    selected,excluded=behaviors(cfg)
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    data=buffer.getvalue()
    target={'model_name_or_path':spec['path'],'adapter_path':spec.get('adapter'),'precision':'nf4',
            'dtype':'bfloat16','num_gpus':1,'use_vllm':False,'ray_gpu_fraction':.45,
            'fschat_template':'qwen3_5_native','trust_remote_code':False}
    api={'model_name_or_path':ALIAS,'fschat_template':'chatgpt','use_vllm':False,'num_gpus':0,'api_variant':variant}
    original=yaml.safe_load((runtime/'configs/method_configs/PAIR_config.yaml').read_text())['default_method_hyperparameters']
    params={**original,'attack_model':api,'judge_model':dict(api),'target_model':target,
            'targets_path':str(resolve(settings['targets'])),
            'target_max_n_tokens':settings['target_max_new_tokens']}
    params.update({k:opts[k] for k in ('n_streams','steps','max_retries','attack_max_n_tokens','judge_max_n_tokens')})
    directory.mkdir(parents=True,exist_ok=True)
    context={'task':'harmbench-official-pair-deepseek-v1','model':model_identity(spec),'parameters':params,
        'behaviors_sha256':hashlib.sha256(data.encode()).hexdigest(),
        'runtime_manifest':digest(runtime/'api_compatibility_manifest.json'),'seed':settings['seed'],
        'upstream_revision':settings['revision'],'api_model':'deepseek-flash','thinking':False,
        'campaign_budget_cny':opts['budget_cny']}
    bind(directory,context)
    (directory/'behaviors.csv').write_text(data)
    (directory/'models.yaml').write_text(yaml.safe_dump({'target':{'model':target,'num_gpus':1}}))
    (directory/'method.yaml').write_text(yaml.safe_dump({'default_method_hyperparameters':params,'run':{}}))
    comparison={k:copy.deepcopy(v) for k,v in params.items() if k!='target_model'}
    for key in ('attack_model','judge_model'):comparison[key].pop('api_variant',None)
    summary={'model_identity':context['model'],'selected_ids':[b['BehaviorID'] for b in selected],
        'excluded_unsupported_ids':excluded,'parameters':comparison,'method':'PAIR','split':settings['split'],
        'protocol':context['task'],'target_max_new_tokens':settings['target_max_new_tokens'],
        'scope':'all_text_behaviors' if not settings['limit'] and not excluded else 'subset',
        'runtime_manifest':context['runtime_manifest'],'api_model':'deepseek-flash','thinking':False,
        'campaign_budget_cny':opts['budget_cny']}
    write_json(directory/'attack_summary.json',summary)
    shutil.copy2(runtime/'api_compatibility_manifest.json',directory/'api_compatibility_manifest.json')
    shutil.copy2(runtime/'api_compatibility.patch',directory/'api_compatibility.patch')
    return selected,params
