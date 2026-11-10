"""Compare two NF4 targets with official PAIR and one shared API budget ledger."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fineval_pipeline.core import bind,digest,fingerprint,load_config,resolve,write_json
from fineval_pipeline.deepseek_api import BudgetLedger
from fineval_pipeline.harmbench_pair_api import build_api_runtime,prepare
from fineval_pipeline.harmbench_completions import install
from fineval_pipeline.gpu_lease import acquire_a100_lease
install()
from fineval_pipeline.harmbench_upstream import judge,run_command


def validate_target_pair(targets,budget):
    variants={x[0] for x in targets}
    if len(targets)!=2 or variants not in ({'base_nf4','dpo'},{'finance','hardened'}):
        raise ValueError('Expected original/DPO or financial/hardened comparison')
    if variants=={'finance','hardened'}:
        if budget is not None and not 0<budget<=20:raise ValueError('Hardening uses its authorized finite ceiling or an explicit unlimited flag')
        if {x[0]:x[2] for x in targets}!={'finance':'M1','hardened':'M2'}:
            raise ValueError('Financial baseline must use M1 and hardened target must use M2')
    elif budget is None or not 0<budget<=25:
        raise ValueError('Legacy comparison has a CNY25 total ceiling')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',nargs=3,action='append',metavar=('VARIANT','CONFIG','MODEL'),required=True)
    parser.add_argument('--budget-root',type=Path,required=True)
    parser.add_argument('--budget-cny',type=float,default=25.)
    parser.add_argument('--no-budget-limit',action='store_true',help='Explicit user-authorized cost tracking without a financial ceiling')
    parser.add_argument('--journal-root',type=Path,help='Separate smoke-test journal; API charges still share the campaign ledger')
    parser.add_argument('--generation-only',action='store_true',help='Integration smoke: generate attacks only, no ASR claim')
    parser.add_argument('--defer-scoring',action='store_true',help='Formal generation on this host; official scoring runs separately on A100')
    args=parser.parse_args()
    if args.no_budget_limit:args.budget_cny=None
    budget_root=args.budget_root.resolve();budget_root.mkdir(parents=True,exist_ok=True)
    journal=args.journal_root.resolve() if args.journal_root else budget_root
    journal.mkdir(parents=True,exist_ok=True)
    validate_target_pair(args.target,args.budget_cny)
    if args.generation_only and args.defer_scoring:raise ValueError('Smoke and formal deferred scoring are different modes')
    lock=(budget_root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=BudgetLedger(budget_root,args.budget_cny)
    cfgs={variant:load_config(path) for variant,path,_ in args.target}
    first=next(iter(cfgs.values()));runtime=build_api_runtime(first)
    targets=[];requested=None
    for variant,config_path,model in args.target:
        cfg=cfgs[variant]
        if cfg['harmbench']['pair'].get('provider')!='deepseek':raise ValueError('DeepSeek provider is not configured')
        if cfg['harmbench']['pair']['budget_cny']!=args.budget_cny:raise ValueError('Budget configuration differs')
        directory=resolve(cfg['output_root'])/model/'harmbench/PAIR'
        rows,params=prepare(cfg,model,variant,directory,runtime)
        if requested is None:requested=rows
        elif rows!=requested:raise ValueError('The two model behavior sets differ')
        targets.append({'variant':variant,'model':model,'config':str(Path(config_path).absolute()),
            'directory':str(directory),'method':str(directory/'method.yaml'),
            'test_cases':str(directory/'test_cases')})
    keyfile=resolve(first['harmbench']['pair']['key_file'])
    if not keyfile.is_file() or keyfile.stat().st_mode&0o077:
        raise ValueError('DeepSeek credential file must exist with owner-only access')
    campaign={'runtime':str(runtime),'budget_root':str(budget_root),'budget_cny':args.budget_cny,
        'behaviors':str(Path(targets[0]['directory'])/'behaviors.csv'),'targets':targets,
        'scope':'two targets; shared API budget including pilot, attack, search judge, retries and uncertain charges',
        'core_pair_sha256':digest(runtime/'baselines/pair/PAIR.py')}
    dispatch=budget_root/'target_dispatch.json'
    if dispatch.exists():campaign['target_dispatch']=json.loads(dispatch.read_text())
    write_json(journal/'campaign.json',campaign)
    env=dict(os.environ,HB_API_BUDGET_ROOT=str(budget_root),HB_API_BUDGET_CNY=str(args.budget_cny),
        HB_DEEPSEEK_KEY_FILE=str(keyfile),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',
        PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONPATH=str(ROOT)+os.pathsep+str(runtime))
    worker='deepseek_pair_distributed_worker.py' if dispatch.exists() else 'deepseek_pair_worker.py'
    if dispatch.exists() and all(endpoint is not None for group in campaign['target_dispatch']['groups'] for endpoint in group.values()):
        env['CUDA_VISIBLE_DEVICES']=''
    command=[str(resolve(first['harmbench']['python'])),'-u',str(ROOT/'scripts'/worker),
             '--campaign',str(journal/'campaign.json')]
    with (journal/'worker.log').open('a') as output:
        process=subprocess.Popen(command,env=env,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        (journal/'worker.pid').write_text(str(process.pid))
        code=process.wait()
    if code:raise RuntimeError('PAIR worker failed; persisted attacks and the shared spending ledger are retained')
    generation=json.loads((journal/'campaign_status.json').read_text())
    all_cases={}
    for target in targets:
        directory=Path(target['directory']);cfg=cfgs[target['variant']]
        individual=Path(target['test_cases'])/'test_cases_individual_behaviors'
        if not individual.exists():all_cases[target['variant']]={};continue
        run_command(cfg,runtime,['merge_test_cases.py','--method_name','PAIR','--save_dir',target['test_cases']],
                    directory/'step1_merge.log')
        all_cases[target['variant']]=json.loads((Path(target['test_cases'])/'test_cases.json').read_text())
    common=set.intersection(*(set(x) for x in all_cases.values()))
    ordered=[row for row in requested if row['BehaviorID'] in common]
    coverage={'expected':len(requested),'common_completed':len(ordered),'full_coverage':len(ordered)==len(requested),
        'completed_by_variant':{k:len(v) for k,v in all_cases.items()},'generation_status':generation['state'],
        'missing_common_ids':[r['BehaviorID'] for r in requested if r['BehaviorID'] not in common],
        'budget':ledger.snapshot()}
    write_json(journal/'coverage.json',coverage)
    if not ordered:raise RuntimeError('No common completed PAIR behaviors; refusing to report zero ASR')
    if args.generation_only:
        write_json(journal/'result.json',{'state':'validation_smoke_generated','coverage':coverage,
            'budget':ledger.snapshot(),'ASR':None,'scope':'integration only; excluded from formal evaluation'})
        return
    if args.defer_scoring:
        write_json(journal/'generation_result.json',{'state':'generated_awaiting_official_scoring',
            'coverage':coverage,'budget':ledger.snapshot(),'ASR':None,
            'scope':'formal paired search; no final ASR until official A100 scoring'})
        return
    gpu_lease=acquire_a100_lease('PAIR final completions and classifier',journal/'scoring_gpu_lease.json')
    for target in targets:
        variant=target['variant'];cfg=cfgs[variant];directory=Path(target['directory'])
        ids=[r['BehaviorID'] for r in ordered]
        scoring=directory/('scoring_'+fingerprint(ids)[:12]);scoring.mkdir(parents=True,exist_ok=True)
        data=io.StringIO(newline='');writer=csv.DictWriter(data,fieldnames=list(ordered[0]));writer.writeheader();writer.writerows(ordered)
        bind(scoring,{'source_attack_run':digest(directory/'run.json'),'ids':ids,
                      'behaviors_sha256':__import__('hashlib').sha256(data.getvalue().encode()).hexdigest()})
        (scoring/'behaviors.csv').write_text(data.getvalue());shutil.copy2(directory/'models.yaml',scoring/'models.yaml')
        write_json(scoring/'test_cases.json',{key:all_cases[variant][key] for key in ids})
        metadata=json.loads((directory/'attack_summary.json').read_text())
        metadata.update(selected_ids=ids,requested_behaviors=len(requested),
            scope='all_text_behaviors' if coverage['full_coverage'] else 'common_completed_subset',
            budget_limited=generation['state']=='budget_exhausted',generation_status=generation['state'])
        write_json(scoring/'attack_summary.json',metadata)
        run_command(cfg,runtime,['generate_completions.py','--model_name','target',
            '--models_config_file',str(scoring/'models.yaml'),'--behaviors_path',str(scoring/'behaviors.csv'),
            '--test_cases_path',str(scoring/'test_cases.json'),'--save_path',str(scoring/'completions.json'),
            '--incremental_update','--max_new_tokens',str(cfg['harmbench']['target_max_new_tokens'])],
            scoring/'step2_completions.log')
        summary=judge(cfg,target['model'],'PAIR',scoring)
        summary.update(campaign_budget=ledger.snapshot(),coverage=coverage,scoring_directory=str(scoring))
        write_json(directory/'summary.json',summary)
        shutil.copy2(scoring/'official_results.json',directory/'official_results.json')
    write_json(journal/'result.json',{'state':'completed' if coverage['full_coverage'] else 'partial_coverage',
                                        'coverage':coverage,'budget':ledger.snapshot()})
    gpu_lease.close()


if __name__=='__main__':main()
