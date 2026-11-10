"""Run the evaluation DAG in isolated processes; blocked stages stay explicit."""
import json
import subprocess
import sys

from .core import ROOT, resolve, write_json


def plan(cfg, config_path):
    prefix = [sys.executable, '-m', 'fineval_pipeline', '--config', str(config_path)]
    jobs = []
    classifier = (resolve(cfg['harmbench']['classifier']['path']).is_dir() and
                  resolve(cfg['harmbench']['classifier']['manifest']).is_file())
    for name in ('M0','M1','M2'):
        registered = bool(cfg['models'][name].get('path'))
        requires = [] if name=='M0' else ['M0.evaluate'] if name=='M1' else ['M0.evaluate','M1.evaluate']
        jobs.append({'id':name+'.evaluate', 'requires':requires, 'available':registered,
            'command':prefix+['run','--models',name,'--tasks','fineval','privacy']})
        if name=='M0':
            continue
        for method in ('GCG','PAIR'):
            attack = name+'.'+method+'.attack'
            jobs.append({'id':attack,'requires':[name+'.evaluate'],'available':registered,
                         'command':prefix+['attack','--model',name,'--method',method]})
            jobs.append({'id':name+'.'+method+'.judge','requires':[attack],'available':registered and classifier,
                         'command':prefix+['judge','--model',name,'--method',method]})
        if name=='M2':
            jobs.append({'id':'M2.agent','requires':['M2.evaluate'],'available':registered,
                         'command':prefix+['run','--models','M2','--tasks','agent']})
    return jobs


def execute(cfg, config_path, dry_run=False):
    root = resolve(cfg['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    jobs = plan(cfg,config_path)
    write_json(root/'plan.json',jobs)
    if dry_run:
        return {'jobs':jobs}
    statuses = {}
    logs = root/'job_logs';logs.mkdir(exist_ok=True)
    for job in jobs:
        if not job['available']:
            statuses[job['id']] = 'blocked_missing_model_or_classifier'
        elif any(statuses.get(dep)!='passed' for dep in job['requires']):
            statuses[job['id']] = 'blocked_dependency'
        else:
            print('START '+job['id'],flush=True)
            with (logs/(job['id']+'.log')).open('w') as log:
                result = subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            statuses[job['id']] = 'passed' if result.returncode==0 else 'failed'
        write_json(root/'suite_status.json',statuses)
    from .report import build
    build(cfg,root)
    return statuses
