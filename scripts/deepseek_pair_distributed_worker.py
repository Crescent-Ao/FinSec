"""Two concurrent behavior pairs across 3 RTX4090 targets and one A100 target."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import copy
import csv
import json
from pathlib import Path
import sys
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fineval_pipeline.core import write_json,append
from fineval_pipeline.deepseek_api import BudgetLedger,BudgetExceeded,APIConfigurationError
from fineval_pipeline.pair_rpc import TargetRPC,TargetRPCError,validate_dispatch
from scripts.deepseek_pair_worker import existing_cases,generate_and_commit


def run_row(methods,targets,params,row,completed):
    result={};errors=[];stop=None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(generate_and_commit,methods[v],row,t['test_cases'],copy.deepcopy(params[v])):v
                 for v,t in targets.items() if row['BehaviorID'] not in completed[v]}
        for future in futures:
            variant=futures[future]
            try:result[variant]=future.result()
            except BudgetExceeded:stop='budget_exhausted'
            except APIConfigurationError:stop='api_credential_or_balance_error'
            except TargetRPCError as exc:
                stop='target_inference_error'
                errors.append({'variant':variant,'behavior_id':row['BehaviorID'],'error_type':type(exc).__name__})
            except Exception as exc:
                errors.append({'variant':variant,'behavior_id':row['BehaviorID'],'error_type':type(exc).__name__})
    return result,errors,stop


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);args=p.parse_args()
    campaign=json.loads(args.campaign.read_text());dispatch=campaign['target_dispatch']
    validate_dispatch(dispatch)
    sys.path.insert(0,campaign['runtime'])
    import ray
    import torch
    from baselines.pair import conversers
    from baselines.pair.PAIR import PAIR
    local_gpu=any(endpoint is None for group in dispatch['groups'] for endpoint in group.values())
    torch.set_num_threads(4);ray.init(num_gpus=int(local_gpu),num_cpus=8,include_dashboard=False,ignore_reinit_error=True)
    ledger=BudgetLedger(campaign['budget_root'],campaign['budget_cny'])
    with Path(campaign['behaviors']).open() as f:rows=list(csv.DictReader(f))
    targets={t['variant']:t for t in campaign['targets']}
    params={v:yaml.safe_load(Path(t['method']).read_text())['default_method_hyperparameters'] for v,t in targets.items()}
    completed={v:existing_cases(PAIR,t['test_cases'],rows) for v,t in targets.items()}
    original_loader=conversers.load_indiv_model;clients=[];groups=[];failures=[];stopped=None
    journal=args.campaign.parent;started=time.monotonic()
    def publish(last=None):
        budget=ledger.snapshot()
        state={'state':stopped or 'generating','expected_behaviors':len(rows),
            'completed_by_variant':{v:len(ids) for v,ids in completed.items()},
            'common_completed':len(set.intersection(*completed.values())),
            'spent_cny':budget['spent_cny'],'committed_upper_cny':budget['committed_upper_cny'],
            'limit_cny':budget['limit_cny'],'failed_behavior_count':len(failures),
            'seconds':time.monotonic()-started,'dispatch':dispatch.get('layout') or ('3xRTX4090 + 1xA100; two behavior pairs' if local_gpu else '3xRTX4090 only; two behavior pairs'),
            'last_behavior_id':last}
        write_json(journal/'campaign_status.json',state);print('PAIR_PROGRESS '+json.dumps(state),flush=True)
        return state
    try:
        for group in dispatch['groups']:
            methods={}
            for variant,endpoint in group.items():
                if endpoint:
                    client=TargetRPC(endpoint,variant,dispatch['token_file'],dispatch['expected'][variant]);clients.append(client)
                    def loader(model_name_or_path,_client=client,**kwargs):
                        if model_name_or_path=='gpt-deepseek-flash':return original_loader(model_name_or_path,**kwargs)
                        return _client
                    conversers.load_indiv_model=loader
                else:conversers.load_indiv_model=original_loader
                methods[variant]=PAIR(**copy.deepcopy(params[variant]))
            groups.append(methods)
        conversers.load_indiv_model=original_loader
        pending_rows=iter([row for row in rows if any(row['BehaviorID'] not in completed[v] for v in targets)])
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            active={}
            def submit(index):
                row=next(pending_rows,None)
                if row:
                    future=pool.submit(run_row,groups[index],targets,params,row,{v:set(ids) for v,ids in completed.items()})
                    active[future]=(index,row['BehaviorID'])
            for index in range(len(groups)):submit(index)
            publish()
            while active:
                finished,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
                if not finished:continue
                for future in finished:
                    index,key=active.pop(future);result,errors,reason=future.result()
                    for variant,bid in result.items():completed[variant].add(bid)
                    append(journal/'target_execution.jsonl',{'behavior_id':key,'group':index,
                        'completed_variants':list(result),'endpoints':dispatch['groups'][index],
                        'time':time.time(),'scope':'target transport only; official PAIR method config unchanged'})
                    failures.extend(errors)
                    if reason:stopped=reason
                    publish(key)
                    if not stopped:submit(index)
        if not stopped:stopped='completed' if all(len(ids)==len(rows) for ids in completed.values()) else 'completed_with_attack_errors'
        publish();write_json(journal/'generation_failures.json',failures)
    finally:
        conversers.load_indiv_model=original_loader
        for methods in groups:
            for method in methods.values():
                actor=method.targetLM.model
                if isinstance(actor,ray.actor.ActorHandle):ray.kill(actor,no_restart=True)
        ray.shutdown()
        for client in clients:
            try:client.shutdown()
            except Exception:pass


if __name__=='__main__':main()
