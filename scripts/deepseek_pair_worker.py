"""Interleave the two official PAIR searches under one durable API budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fineval_pipeline.core import write_json
from fineval_pipeline.deepseek_api import BudgetLedger,BudgetExceeded,APIConfigurationError


def existing_cases(method,directory,rows):
    result=set()
    for row in rows:
        key=row['BehaviorID'];path=Path(method.get_output_file_path(directory,key,'test_cases'))
        if not path.exists():continue
        obj=json.loads(path.read_text())
        if set(obj)!={key} or not obj[key] or not all(isinstance(x,str) and x.strip() for x in obj[key]):
            raise ValueError('Invalid saved PAIR checkpoint: '+key)
        result.add(key)
    return result


def generate_and_commit(method,row,directory,params):
    cases,logs=method.generate_test_cases([row],verbose=False)
    key=row['BehaviorID'];root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pending-',dir=root) as temp:
        method.save_test_cases(temp,cases,logs,method_config=params)
        pending=Path(method.get_output_file_path(temp,key,'test_cases')).parent
        destination=Path(method.get_output_file_path(root,key,'test_cases')).parent
        destination.mkdir(parents=True,exist_ok=True)
        for path in sorted(pending.iterdir(),key=lambda x:x.name.startswith('test_cases')):
            os.replace(path,destination/path.name)
    return key


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();campaign=json.loads(args.campaign.read_text())
    sys.path.insert(0,campaign['runtime'])
    import ray
    from baselines.pair.PAIR import PAIR
    import torch
    torch.set_num_threads(4)
    ray.init(num_gpus=1,num_cpus=8,include_dashboard=False,ignore_reinit_error=True)
    ledger=BudgetLedger(campaign['budget_root'],campaign['budget_cny'])
    with Path(campaign['behaviors']).open() as f:rows=list(csv.DictReader(f))
    methods={};completed={};params={};failures=[];stopped=None
    journal=args.campaign.parent
    started=time.monotonic();status_path=journal/'campaign_status.json'
    try:
        for target in campaign['targets']:
            variant=target['variant'];params[variant]=yaml.safe_load(Path(target['method']).read_text())['default_method_hyperparameters']
            methods[variant]=PAIR(**params[variant])
            completed[variant]=existing_cases(methods[variant],target['test_cases'],rows)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for row in rows:
                key=row['BehaviorID'];futures={}
                for target in campaign['targets']:
                    variant=target['variant']
                    if key not in completed[variant]:
                        futures[pool.submit(generate_and_commit,methods[variant],row,target['test_cases'],params[variant])]=variant
                for future in as_completed(futures):
                    variant=futures[future]
                    try:completed[variant].add(future.result())
                    except BudgetExceeded:stopped='budget_exhausted'
                    except APIConfigurationError as exc:
                        stopped='api_credential_or_balance_error';failures.append({'variant':variant,'behavior_id':key,'error':str(exc)})
                    except Exception as exc:
                        failures.append({'variant':variant,'behavior_id':key,'error_type':type(exc).__name__,
                            'error':'official PAIR behavior generation failed; no zero-ASR credit'})
                budget=ledger.snapshot()
                state={'state':stopped or 'generating','expected_behaviors':len(rows),
                    'completed_by_variant':{k:len(v) for k,v in completed.items()},
                    'common_completed':len(set.intersection(*completed.values())),
                    'spent_cny':budget['spent_cny'],'committed_upper_cny':budget['committed_upper_cny'],
                    'limit_cny':budget['limit_cny'],'last_behavior_id':key,'failed_behavior_count':len(failures),
                    'seconds':time.monotonic()-started}
                write_json(status_path,state);print('PAIR_PROGRESS '+json.dumps(state),flush=True)
                if stopped:break
        state['state']=stopped or ('completed' if all(len(v)==len(rows) for v in completed.values()) else 'completed_with_attack_errors')
        write_json(status_path,state);write_json(journal/'generation_failures.json',failures)
    finally:
        for method in methods.values():
            actor=getattr(getattr(method,'targetLM',None),'model',None)
            if isinstance(actor,ray.actor.ActorHandle):ray.kill(actor,no_restart=True)
        ray.shutdown()


if __name__=='__main__':main()
