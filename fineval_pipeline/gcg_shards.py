"""Validate and publish independently generated GCG behavior shards."""
import argparse
import copy
import fcntl
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time

from .core import digest, fingerprint, resolve, write_json

PACKAGES=('torch','transformers','triton','bitsandbytes','peft','flash-linear-attention','fla-core')


def versions():return {name:importlib.metadata.version(name) for name in PACKAGES}


def parameters(value,targets_sha):
    result=copy.deepcopy(value)
    result.pop('target_model',None);result.pop('dependencies',None)
    result['targets_path']='sha256:'+targets_sha
    return result


def protocol(model,metadata,context,targets_sha,software):
    return {'model_identity':model,'selected_ids':metadata['selected_ids'],
        'parameters':parameters(metadata['parameters'],targets_sha),'targets_sha256':targets_sha,
        'behaviors_sha256':context['behaviors_sha256'],'runtime_manifest':metadata['runtime_manifest'],
        'upstream_revision':context['upstream_revision'],'seed':context['seed'],
        'seed_protocol':'sha256(base_seed:BehaviorID) first 32 bits','software':software}


def balanced_shards(items,weights,count):
    if count<1 or len(set(items))!=len(items):raise ValueError('Invalid GCG shard inputs')
    groups=[[] for _ in range(count)];loads=[0.]*count
    for key in sorted(items,key=lambda x:(-weights[x],x)):
        index=min(range(count),key=lambda i:(loads[i],i));groups[index].append(key);loads[index]+=weights[key]
    return groups,loads


def initialize(receipt,expected,assignments,deadline_seconds=12*3600,destination=None):
    receipt=Path(receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    flat=[x for group in assignments.values() for x in group]
    if len(flat)!=len(set(flat)) or set(flat)!=set(expected['selected_ids']):
        raise ValueError('GCG assignments are duplicated or incomplete')
    with receipt.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            state=json.loads(receipt.read_text())
            if state['expected']!=expected or state['assignments']!=assignments:
                raise ValueError('Refusing to replace a different distributed GCG experiment')
            return state
        state={'state':'running','created_epoch':time.time(),'deadline_epoch':time.time()+deadline_seconds,
            'expected':expected,'assignments':assignments,'validated_behaviors':0,
            'destination':str(destination) if destination else None,
            'origin':'40903 RTX4090 behavior shards; target completions and official classifier remain on A100'}
        write_json(receipt,state);return state


def finalize(receipt,destination,reason='workers_finished',dry_run=False):
    receipt=Path(receipt)
    with receipt.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=json.loads(receipt.read_text())
        if state['state']!='running':return state
        destination=Path(destination or state['destination'])
        export_file=receipt.parent/'incoming/export.json'
        exported=json.loads(export_file.read_text()) if export_file.exists() else None
        records=[]
        if exported:
            if exported['protocol']!=state['expected']:raise ValueError('Cross-host GCG model/data/budget/runtime/software mismatch')
            records=exported['records']
            adapters=exported.get('source_adapters')
            if adapters is not None:
                if set(adapters)!=set(state['assignments']):
                    raise ValueError('Every heterogeneous shard needs its explicit source adapter')
                if any(not isinstance(item.get('path'),str) or not item['path']
                       or item.get('model_identity')!=state['expected']['model_identity']
                       for item in adapters.values()):
                    raise ValueError('Heterogeneous GCG shards have another model identity')
        seen=set();validated=[]
        for record in records:
            key=record['behavior_id'];shard=str(record['shard'])
            if key in seen or key not in state['assignments'].get(shard,[]):
                raise ValueError('Duplicate or wrongly assigned imported behavior')
            seen.add(key)
            relative=PurePosixPath(record['relative_dir'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid GCG import path')
            source=receipt.parent/'incoming'/relative
            if set(record['files'])!={'test_cases.json','logs.json','method_config.json'}:
                raise ValueError('Incomplete official behavior checkpoint')
            for name,sha in record['files'].items():
                file=source/name
                if file.is_symlink() or not file.is_file() or digest(file)!=sha:
                    raise ValueError('Imported GCG artifact is missing or changed: '+key)
            cases=json.loads((source/'test_cases.json').read_text())
            if set(cases)!={key} or not cases[key] or not all(isinstance(x,str) and x.strip() for x in cases[key]):
                raise ValueError('Invalid imported GCG test case: '+key)
            config=json.loads((source/'method_config.json').read_text())
            if parameters(config,state['expected']['targets_sha256'])!=state['expected']['parameters']:
                raise ValueError('Imported behavior attack parameters differ')
            target=config['target_model']
            source_adapter=(exported['source_adapters'][shard]['path']
                            if exported.get('source_adapters') is not None else exported['source_adapter_path'])
            if target.get('precision')!='nf4' or target.get('adapter_path')!=source_adapter:
                raise ValueError('Imported behavior used a different target/precision')
            validated.append((key,source,record))
        if dry_run:
            return {**state,'preview_validated_behaviors':len(validated)}
        # Validate everything before making any behavior visible to upstream's skip logic.
        for key,source,record in validated:
            folder=destination/key;folder.mkdir(parents=True,exist_ok=True)
            for name in ('logs.json','method_config.json','test_cases.json'):
                path=folder/name
                if path.exists():
                    if digest(path)!=record['files'][name]:raise ValueError('Refusing to overwrite an existing GCG behavior')
                    continue
                temp=folder/(name+'.importing');shutil.copy2(source/name,temp);temp.replace(path)
        count=len(validated)
        state.update(state='ready_full' if count==len(state['expected']['selected_ids']) else 'ready_partial',
            validated_behaviors=count,missing_ids=[x for x in state['expected']['selected_ids'] if x not in seen],
            finalized_epoch=time.time(),reason=reason,
            hardware=exported.get('hardware',{}) if exported else {},
            source_adapters=exported.get('source_adapters') if exported else None,
            exported_manifest_sha256=digest(export_file) if export_file.exists() else None)
        write_json(receipt,state)
        write_json(destination.parent.parent/'distributed_search_provenance.json',state)
        return state


def wait_for_import(cfg,name):
    options=cfg['harmbench'].get('distributed_gcg_import')
    if not options or name!=options['model']:return None
    spec=cfg['models'][name]
    if str(resolve(spec.get('adapter')))!=options['adapter_path'] or str(resolve(cfg['output_root']))!=options['output_root']:
        return None
    receipt=resolve(options['receipt']);destination=resolve(cfg['output_root'])/name/'harmbench/GCG/test_cases/test_cases_individual_behaviors'
    while True:
        state=json.loads(receipt.read_text())
        if state['state']!='running':return state
        if time.time()>=state['deadline_epoch']:
            return finalize(receipt,destination,'deadline: finish the remaining behaviors on A100')
        print('Waiting for RTX4090 GCG shards; source artifacts remain checkpointed.',flush=True)
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--initialize-from-comparison',type=Path)
    parser.add_argument('--initialize-from-attack',type=Path)
    parser.add_argument('--source-plan',type=Path)
    args=parser.parse_args()
    if args.initialize_from_attack:
        attack=args.initialize_from_attack
        context=json.loads((attack/'run.json').read_text())['context']
        metadata=json.loads((attack/'attack_summary.json').read_text())
        expected=protocol(context['model'],metadata,context,digest(context['parameters']['targets_path']),versions())
        plan=json.loads(args.source_plan.read_text())
        if plan['protocol']!=expected:raise ValueError('RTX4090 GRPO model/data/runtime/software differs from A100')
        result=initialize(args.receipt,expected,plan['assignments'],deadline_seconds=7*86400,destination=args.destination)
    elif args.initialize_from_comparison:
        comparison=args.initialize_from_comparison
        base=comparison/'base_nf4/M0/harmbench/GCG'
        context=json.loads((base/'run.json').read_text())['context']
        metadata=json.loads((base/'attack_summary.json').read_text())
        identity=json.loads((comparison/'dpo/M2/fineval/summary.json').read_text())['model_identity']
        expected=protocol(identity,metadata,context,digest(context['parameters']['targets_path']),versions())
        plan=json.loads(args.source_plan.read_text())
        if plan['protocol']!=expected:raise ValueError('40903 software/model/data/attack protocol differs from the A100 baseline')
        result=initialize(args.receipt,expected,plan['assignments'],destination=args.destination)
    elif args.finalize or args.verify_only:
        result=finalize(args.receipt,args.destination,dry_run=args.verify_only)
    else:
        with args.receipt.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);result=json.loads(args.receipt.read_text())
            exported=args.receipt.parent/'incoming/export.json'
            if result['state']=='running' and exported.exists():
                data=json.loads(exported.read_text())
                if data['protocol']!=result['expected']:raise ValueError('Staged shard protocol differs')
                result['staged_behaviors']=len(data['records'])
                result['source_completed_by_shard']={key:sum(r['shard']==key for r in data['records']) for key in result['assignments']}
                write_json(args.receipt,result)
    print(json.dumps({'state':result['state'],'validated_behaviors':result['validated_behaviors'],
                      'staged_behaviors':result.get('staged_behaviors',0),
                      'preview_validated_behaviors':result.get('preview_validated_behaviors')}))


if __name__=='__main__':main()
