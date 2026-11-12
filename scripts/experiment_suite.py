"""Portable five-stage evaluation planning; no inference without --execute."""
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = {'fineval', 'privacy', 'agent', 'gcg', 'pair'}


def resolve(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else ROOT / path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(path, tasks=None):
    config = json.loads(resolve(path).read_text())
    stages = config['stages']
    ids = [stage['id'] for stage in stages]
    if len(ids) != len(set(ids)) or not ids or ids[0] != 'M0' or 'M1' not in ids:
        raise ValueError('Unique stage IDs must start with M0 and include M1')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in ids):
        raise ValueError('Stage IDs must be safe directory names')
    if stages[0].get('adapter') or any(not s.get('adapter') for s in stages[1:]):
        raise ValueError('M0 is the unadapted baseline; every later stage needs its own adapter')
    if len({str(resolve(s['adapter'])) for s in stages[1:]}) != len(stages) - 1:
        raise ValueError('Use distinct adapter paths for distinct stages')
    tasks = tasks if tasks is not None else config.get('tasks', ['fineval'])
    if not tasks or len(tasks) != len(set(tasks)) or not set(tasks) <= TASKS:
        raise ValueError('Choose unique tasks from fineval, privacy, agent, gcg, pair')
    pipeline = yaml.safe_load(resolve(config['pipeline_config']).read_text())
    if pipeline.get('harmbench', {}).get('pair', {}).get('provider') == 'deepseek' and 'pair' in tasks:
        raise ValueError('API PAIR requires the separate campaign and explicit budget; use a local attacker here')
    output = resolve(config['output_root'])
    if output == ROOT or output in (ROOT / 'configs', ROOT / 'scripts', ROOT / 'docs'):
        raise ValueError('Choose a dedicated experiment output directory')
    pipeline['output_root'] = str(output)
    pipeline['hardware'] = {'cuda_visible_devices': str(config.get('gpus', '0'))}
    pipeline['experiment_protocol'] = {'required_target_precision': 'nf4', 'adapter_loading': 'unmerged'}
    models = {}
    for stage in stages:
        spec = dict(pipeline.get('models', {}).get(stage['id'], {}))
        spec.update(path=str(resolve(config['model_dir'])), precision='nf4', system_prompt='', general_instruction_model=True)
        for key in ('adapter', 'evaluation_export', 'stage', 'parent', 'safety_manifest'):
            spec.pop(key, None)
        if stage.get('adapter'):
            spec['adapter'] = str(resolve(stage['adapter']))
        models[stage['id']] = spec
    pipeline['models'] = models
    if 'agent' in tasks:
        pipeline['agent']['knowledge'] = str(resolve(pipeline['agent'].get('knowledge', 'knowledge/crypto_rules.json')))
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = output / 'suite.yaml'
    jobs = []
    specifications = []
    python = str(resolve(config['python'])) if config.get('python') else sys.executable
    for stage in stages:
        name = stage['id']
        prefix = [python, '-m', 'fineval_pipeline', '--config', str(cfg_path)]
        for task in tasks:
            commands = []
            if task in ('fineval', 'privacy'):
                commands.append(prefix + ['run', '--models', name, '--tasks', task])
            elif task in ('gcg', 'pair'):
                commands.extend(prefix + [action, '--model', name, '--method', task.upper()] for action in ('attack', 'judge'))
            else:
                spec_path = output / name / 'agent_spec.json'
                specifications.append((spec_path, models[name]))
                commands.append([config.get('node', 'node'), str(ROOT / 'pi_crypto_agent/evaluate_described_4096_pi_only.mjs'),
                    '--spec', str(spec_path), '--cases', str(resolve(pipeline['agent']['cases'])),
                    '--knowledge', str(resolve(pipeline['agent']['knowledge'])), '--python', python,
                    '--output', str(output / name / 'agent')])
            for command in commands:
                jobs.append({'stage': name, 'task': task, 'command': command})
    plan = {'schema_version': 1, 'baseline': 'M0', 'stages': stages, 'tasks': tasks,
            'config_sha256': hashlib.sha256(json.dumps(pipeline, sort_keys=True).encode()).hexdigest(),
            'output_root': str(output), 'gpus': str(config.get('gpus', '0')), 'jobs': jobs,
            'note': 'Planning writes configurations only. All target models use the same NF4 base and unmerged adapters.'}
    previous = output / 'plan.json'
    if previous.exists() and json.loads(previous.read_text()) != plan:
        raise ValueError('Suite configuration changed; choose a new output_root')
    cfg_path.write_text(yaml.safe_dump(pipeline, allow_unicode=True, sort_keys=False))
    for spec_path, spec in specifications:
        write(spec_path, spec)
    write(output / 'plan.json', plan)
    return plan


def collect(plan):
    output = Path(plan['output_root'])
    result = {'baseline': 'M0', 'stages': [], 'missing': []}
    baseline_finance = None
    for stage in plan['stages']:
        row = {'id': stage['id'], 'label': stage['label']}
        for task in plan['tasks']:
            relative = {'agent': 'agent/pi.summary.json', 'gcg': 'harmbench/GCG/summary.json',
                        'pair': 'harmbench/PAIR/summary.json'}.get(task, task + '/summary.json')
            path = output / stage['id'] / relative
            if not path.exists():
                row[task] = None
                result['missing'].append(str(path.relative_to(output)))
                continue
            row[task] = json.loads(path.read_text())
        if row.get('fineval'):
            f = row['fineval']
            if stage['id'] == 'M0':
                baseline_finance = f
            elif baseline_finance:
                for key in ('task', 'data_sha256', 'examples', 'limit_per_subject'):
                    if f[key] != baseline_finance[key]:
                        raise ValueError('FinEval protocols differ: ' + key)
                row['fineval_delta_pp_vs_base'] = 100 * (f['micro_accuracy'] - baseline_finance['micro_accuracy'])
        result['stages'].append(row)
    result['complete'] = not result['missing']
    write(output / 'comparison.json', result)
    lines = ['# Local evaluation comparison', '', 'Baseline: M0 (NF4). Missing outputs remain —.', '',
             '| Stage | FinEval | Δ vs base (pp) | GCG ASR | PAIR ASR | Pi Precision | Pi Recall | Pi F1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def percent(obj, key):
        return '—' if obj is None else f'{obj[key] * 100:.2f}%'
    for row in result['stages']:
        f, agent = row.get('fineval'), row.get('agent')
        delta = row.get('fineval_delta_pp_vs_base')
        cells = [row['label'], percent(f, 'micro_accuracy'), f'{delta:+.2f}' if delta is not None else '—',
                 percent(row.get('gcg'), 'asr'), percent(row.get('pair'), 'asr')]
        cells += [percent(agent, key) for key in ('precision', 'recall', 'f1')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Attack rates are raw per-run results: compare BehaviorID sets and search budgets before interpreting differences.',
              'Full MIA AUC/TPR/FPR, model identities and protocol metadata are retained in comparison.json.',
              'A partial run is not a completed reproduction of the published competition results.', '']
    if 'privacy' in plan['tasks']:
        lines += ['## Membership inference', '', '| Stage | Attack | ROC-AUC | TPR@1%FPR | Actual FPR |',
                  '|---|---|---:|---:|---:|']
        for row in result['stages']:
            for name, values in (row.get('privacy') or {}).get('attacks', {}).items():
                lines.append(f"| {row['label']} | {name} | {values['roc_auc']:.6f} | {values['tpr_at_1pct_fpr'] * 100:.4f}% | {values['achieved_fpr'] * 100:.4f}% |")
    (output / 'comparison.md').write_text('\n'.join(lines))
    return result


def run(args):
    plan = prepare(args.config, args.tasks)
    output = Path(plan['output_root'])
    if args.execute:
        statuses = []
        for job in plan['jobs']:
            status = {**job, 'state': 'running'}
            statuses.append(status)
            write(output / 'status.json', statuses)
            completed = subprocess.run(job['command'], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=plan['gpus']))
            status.update(state='completed' if completed.returncode == 0 else 'failed', returncode=completed.returncode)
            write(output / 'status.json', statuses)
            if completed.returncode:
                raise SystemExit(completed.returncode)
    if args.collect or args.execute:
        result = collect(plan)
        print(json.dumps({'comparison': str(output / 'comparison.md'), 'complete': result['complete'], 'missing': result['missing']}, ensure_ascii=False, indent=2))
        if args.execute and not result['complete']:
            raise SystemExit(2)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
