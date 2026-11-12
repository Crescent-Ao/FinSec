import argparse
import importlib.util
import json
import subprocess
import os
from pathlib import Path

from .core import Backend, ROOT, load_config, model_identity, model_spec, parameter_count, resolve, write_json


def preflight(cfg):
    from .privacy import membership_data
    from .harmbench import upstream, behaviors
    checks = {}
    count = parameter_count(resolve(cfg['models']['M0']['path']))
    checks['base_model'] = {'parameters': count, 'within_10B': 0 < count <= 10_000_000_000,
                            'general_instruction_declared': cfg['models']['M0'].get('general_instruction_model') is True}
    if not checks['base_model']['within_10B'] or not checks['base_model']['general_instruction_declared']:
        raise ValueError('Base model eligibility not established')
    checks['models'] = {}
    for name in cfg['models']:
        try:
            spec = model_spec(cfg, name)
            checks['models'][name] = {'weights_exist': Path(spec['path']).is_dir(),
                'adapter_exists': Path(spec['adapter']).is_dir() if spec.get('adapter') else None}
        except ValueError as error:
            checks['models'][name] = {'status': 'not_registered','reason':str(error)}
    from scripts.eval_fineval import load_rows
    from .finance import verify_reference
    verify_reference(cfg)
    questions, sha = load_rows(resolve(cfg['fineval']['data_dir']), 0)
    checks['fineval'] = {'labeled_val_questions': len(questions), 'subjects': len({r['subject'] for r in questions}), 'sha256': sha}
    selected, _ = membership_data(cfg)
    checks['privacy'] = {'members': sum(y==1 for r,y in selected), 'nonmembers': sum(y==0 for r,y in selected),
                         'training_snapshot_and_isolation': 'passed'}
    source = upstream(cfg)
    selected, excluded = behaviors(cfg)
    checks['harmbench'] = {'revision': cfg['harmbench']['revision'], 'selected': len(selected), 'excluded': len(excluded),
        'official_classifier_exists': (resolve(cfg['harmbench']['classifier']['path']).is_dir() and
                                       resolve(cfg['harmbench']['classifier']['manifest']).is_file()),
        'copyright_dependencies': {m: importlib.util.find_spec(m) is not None for m in ('spacy','datasketch','en_core_web_sm')}}
    from .core import rows
    checks['agent'] = {'authored_smoke_cases':len(rows(resolve(cfg['agent']['cases']))),
                       'knowledge_rules':len(json.loads(resolve(cfg['agent']['knowledge']).read_text())),
                       'external_core_model_calls':0}
    checks['pending'] = []
    if checks['models'].get('M2',{}).get('status')=='not_registered':
        checks['pending'].append('Register actual safety-enhanced M2; M1 is not silently reused')
    if not checks['harmbench']['official_classifier_exists']:
        checks['pending'].append('Download official cais/HarmBench-Llama-2-13b-cls classifier')
    if not all(checks['harmbench']['copyright_dependencies'].values()):
        checks['pending'].append('Copyright scoring requires spacy + en_core_web_sm + datasketch')
    if cfg['privacy'].get('limit_per_class') or len(selected)<320:
        checks['pending'].append('Smoke limits enabled; not a full evaluation')
    return checks


def main():
    parser = argparse.ArgumentParser(description='Four-stage finance competition evaluation pipeline')
    active_config = ROOT/'reports/latest_competition_evaluation_config.txt'
    default_config = active_config.read_text().strip() if active_config.exists() else 'configs/competition.yaml'
    parser.add_argument('--config', default=default_config)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('preflight')
    run = sub.add_parser('run')
    run.add_argument('--models', nargs='+', default=['M0','M1','M2'])
    run.add_argument('--tasks', nargs='+', choices=['fineval','privacy','agent'], default=['fineval','privacy','agent'])
    attack = sub.add_parser('attack')
    attack.add_argument('--model', required=True, help='A model key registered in the configuration')
    attack.add_argument('--method', choices=['GCG','PAIR'], required=True)
    judge = sub.add_parser('judge')
    judge.add_argument('--model', required=True, help='A model key registered in the configuration')
    judge.add_argument('--method', choices=['GCG','PAIR'], required=True)
    sub.add_parser('report')
    sub.add_parser('plan')
    sub.add_parser('suite')
    export=sub.add_parser('export-model')
    export.add_argument('--model',required=True)
    safety = sub.add_parser('train-safety')
    safety.add_argument('--execute', action='store_true', help='Without this flag, write the reviewed training plan only')
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    if args.command in ('attack','judge') and args.model not in cfg['models']:
        parser.error('Target model is not registered in the configuration')
    if cfg['harmbench']['pair'].get('provider')=='deepseek' and (
            args.command=='suite' or args.command in ('attack','judge') and args.method=='PAIR'):
        parser.error('DeepSeek PAIR uses the paired campaign to enforce the shared CNY25 limit. '
                     'Use scripts/run_deepseek_pair_campaign.py or the comparison runner with --variants base_nf4 dpo.')
    if cfg.get('hardware',{}).get('cuda_visible_devices') is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=str(cfg['hardware']['cuda_visible_devices'])
    root = resolve(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    if args.command=='preflight':
        result = preflight(cfg); write_json(root/'preflight.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return
    if args.command in ('plan','suite'):
        from .suite import execute
        status=execute(cfg,resolve(args.config),args.command=='plan')
        print(json.dumps(status,ensure_ascii=False,indent=2))
        if args.command=='suite' and any(s!='passed' for s in status.values()):
            raise SystemExit(2)
        return
    if args.command=='report':
        from .report import build
        result = build(cfg,root); print(json.dumps(result['stage_status'],indent=2)); return
    if args.command=='train-safety':
        from .safety import build_plan
        print(json.dumps(build_plan(cfg,args.execute),ensure_ascii=False,indent=2)); return
    if args.command=='export-model':
        from .export_model import export
        print(json.dumps(export(cfg,args.model),ensure_ascii=False,indent=2)); return
    if args.command in ('attack','judge') and cfg['harmbench'].get('engine')=='upstream':
        from .harmbench_completions import install
        install()
        imported=None
        if args.command=='attack' and args.method=='GCG':
            from .gcg_shards import wait_for_import
            imported=wait_for_import(cfg,args.model)
        from .harmbench_upstream import attack,judge
        directory=root/args.model/'harmbench'/args.method
        if args.command=='attack':
            attack(cfg,args.model,args.method,directory)
            if imported:
                path=directory/'attack_summary.json';summary=json.loads(path.read_text())
                summary['distributed_search']={'imported_behaviors':imported['validated_behaviors'],
                    'state':imported['state'],'origin':imported['origin'],'hardware':imported.get('hardware',{}),
                    'provenance_file':str(directory/'distributed_search_provenance.json'),
                    'note':'Search may differ numerically across GPU architectures; NF4 target completions and official scoring run on A100.'}
                write_json(path,summary)
        else:print(json.dumps(judge(cfg,args.model,args.method,directory),indent=2))
        return
    if args.command=='judge':
        from .harmbench import score
        print(json.dumps(score(cfg,args.method,root/args.model/'harmbench'/args.method),indent=2)); return
    import torch
    torch.manual_seed(cfg['seed'])
    names = args.models if args.command=='run' else [args.model]
    # Resolve all requested models first so an unset M2 cannot produce a partial "all" run.
    specs = {n:model_spec(cfg,n) for n in names}
    for name,spec in specs.items():
        identity = model_identity(spec)
        backend = Backend(spec)
        backend.identity = identity
        if args.command=='attack':
            from .harmbench import attack
            auxiliary = {}
            if args.method=='PAIR':
                for other in {cfg['harmbench']['pair']['attacker'],cfg['harmbench']['pair']['search_judge']}:
                    if other==name:
                        auxiliary[other] = backend
                    else:
                        other_spec = model_spec(cfg,other)
                        auxiliary[other] = Backend(other_spec)
                        auxiliary[other].identity = model_identity(other_spec)
            attack(cfg,name,args.method,backend,identity,auxiliary,root/name/'harmbench'/args.method)
            print('Attack generation complete. Official ASR requires the separate judge command.')
        else:
            for task in args.tasks:
                if task=='fineval':
                    from .finance import evaluate
                    evaluate(cfg,name,backend,identity,root/name/task)
                elif task=='privacy':
                    from .privacy import evaluate
                    evaluate(cfg,name,backend,identity,root/name/task,root)
                else:
                    from .agent import evaluate
                    evaluate(cfg,name,backend,identity,root/name/task)
        del backend
        import gc
        gc.collect(); torch.cuda.empty_cache()
    if args.command=='run':
        from .report import build
        build(cfg,root)


if __name__=='__main__':
    main()
