"""A100 NF4 safety SFT after MIA, then the frozen financial capability gate."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import digest, write_json
from fineval_pipeline.gpu_lease import acquire_a100_lease
from scripts.run_stage2_posttraining import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--phase-name', default='phase1')
    parser.add_argument('--save-steps', type=int, default=50)
    parser.add_argument('--capacity-batches', type=int, nargs='+', default=[24, 16, 12, 8])
    parser.add_argument('--algorithm', choices=['sft', 'dpo'], default='sft')
    parser.add_argument('--effective-batch', type=int, default=48)
    args = parser.parse_args()
    if args.effective_batch <= 0 or any(batch <= 0 or args.effective_batch % batch for batch in args.capacity_batches):
        raise ValueError('Capacity batches must divide the preserved effective batch')
    stage = args.algorithm
    run = args.run.absolute()
    singleton = (run / '.training_supervisor.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = run / 'status.json'
    status = {'state': 'waiting_for_MIA', 'supervisor_pid': os.getpid(), 'stages': {}}
    def publish(**kwargs):
        status.update(kwargs, updated_epoch=time.time())
        write_json(status_path, status)
    def execute(command, log, stage, env):
        with log.open('a') as output:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            publish(state=stage, child_pid=child.pid, command=command)
            while True:
                try:
                    return child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    publish()
    mia = ROOT / 'outputs/stage1_membership_nf4_base_finance_dpo_20261108_v1/status.json'
    while True:
        state = json.loads(mia.read_text())
        if state['state'] == 'completed':
            break
        if state['state'] == 'failed':
            raise RuntimeError('MIA failed; preserve it before scheduling new training')
        publish(mia_active_model=state.get('active_model'))
        time.sleep(20)
    lease = acquire_a100_lease('Hardening safety ' + stage + ' ' + args.phase_name, run / 'gpu_lease.json')
    try:
        manifest = json.loads((run / 'data_manifest.json').read_text())
        for name, record in manifest['files'].items():
            if digest(run / 'data' / name) != record['sha256']:
                raise ValueError('Frozen hardening data changed: ' + name)
        while True:
            used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True).strip())
            if used < 1024:
                break
            publish(state='waiting_for_A100_release', gpu_used_mib=used)
            time.sleep(15)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', WANDB_MODE='disabled',
            PYTORCH_ALLOC_CONF='expandable_segments:True', STAGE2_RUN=str(run),
            STAGE2_STAGE=stage, STAGE2_PARENT=str(args.parent))
        selected = None
        trials = []
        for batch in args.capacity_batches:
            trial = run / 'capacity' / ('micro' + str(batch))
            trial.mkdir(parents=True, exist_ok=True)
            probe = configuration(stage, args.parent, run, 1, 'nf4')
            stress = trial / 'stress.jsonl'
            longest = (run / 'data/stress.jsonl').open().readline()
            stress.write_text(longest * max(64, 2 * batch))
            probe['arguments'].update(dataset=[str(stress)], val_dataset=[],
                max_steps=2, learning_rate=0., warmup_ratio=0.,
                per_device_train_batch_size=batch, gradient_accumulation_steps=1,
                eval_strategy='no', save_strategy='no', report_to=[], logging_steps=1,
                callbacks=['a100_memory_probe'], external_plugins=[str(ROOT / 'scripts/a100_memory_probe.py')],
                output_dir=str(trial / 'trainer'))
            config = trial / 'config.json'
            write_json(config, probe)
            metrics = trial / 'metrics.json'
            probe_env = dict(env, A100_PROFILE_METRICS=str(metrics))
            code = execute([str(ROOT / '.venv-train-fast/bin/python'),
                str(ROOT / 'scripts/stage2_swift_entry.py'), str(config)], trial / 'run.log',
                'capacity_probe', probe_env)
            if code:
                if 'out of memory' not in (trial / 'run.log').read_text(errors='replace').lower():
                    raise RuntimeError('Capacity probe failed: ' + str(trial / 'run.log'))
                trials.append({'batch': batch, 'state': 'out_of_memory'})
                continue
            metric = json.loads(metrics.read_text())
            if not metric['is_loaded_in_4bit']:
                raise ValueError('Capacity probe did not preserve NF4')
            trials.append({'batch': batch, 'state': 'passed', 'metrics': metric})
            if metric['peak_reserved_GiB'] <= 77:
                selected = batch
                break
        if selected is None:
            raise RuntimeError('No safe batch found for hardening SFT')
        write_json(run / 'capacity_plan.json', {'micro_batch': selected,
            'effective_batch': args.effective_batch,
            'gradient_accumulation': args.effective_batch // selected, 'trials': trials})
        config = configuration(stage, args.parent, run, 1, 'nf4')
        config['arguments'].update(learning_rate=args.learning_rate, per_device_train_batch_size=selected,
            gradient_accumulation_steps=args.effective_batch // selected, logging_steps=1,
            save_steps=args.save_steps, save_total_limit=3, warmup_ratio=.03)
        existing = [p for p in (run / stage).glob('**/checkpoint-*/trainer_state.json')
                    if (p.parent / 'optimizer.pt').exists() and (p.parent / 'scheduler.pt').exists()]
        if existing:
            latest = max(existing, key=lambda p: json.loads(p.read_text())['global_step'])
            config['arguments']['resume_from_checkpoint'] = str(latest.parent)
        path = run / (stage + '_config.json')
        write_json(path, config)
        status['stages'][stage] = {'status': 'running', 'parent': str(args.parent)}
        code = execute([str(ROOT / '.venv-train-fast/bin/python'),
            str(ROOT / 'scripts/stage2_swift_entry.py'), str(path)], run / (stage + '.log'),
            'training_safety_' + stage, env)
        if code:
            publish(state='training_failed', returncode=code)
            return
        checkpoints = list((run / stage).glob('**/checkpoint-*/trainer_state.json'))
        if not checkpoints:
            raise RuntimeError('Safety SFT completed without a checkpoint')
        state = max(checkpoints, key=lambda p: json.loads(p.read_text())['global_step'])
        checkpoint = state.parent
        if not (checkpoint / 'adapter_model.safetensors').is_file():
            raise ValueError('Saved safety adapter is missing')
        status['stages'][stage] = {'status': 'completed', 'checkpoint': str(checkpoint)}
        lineage = run / 'lineage.json'
        write_json(lineage, {'parent': 'M1', 'nonmembers_excluded': True,
            'direct_parent_checkpoint': str(args.parent), 'checkpoint': str(checkpoint),
            'training_manifest_sha256': digest(run / 'data_manifest.json'),
            'stage': 'safety adversarial ' + stage.upper() + ' ' + args.phase_name + '; crypto GRPO not resumed'})
        cfg = yaml.safe_load((ROOT / 'outputs/nf4_four_stage_comparison_gcg_optimized_20261107_v2/dpo.yaml').read_text())
        cfg['models']['M2'].update(adapter=str(checkpoint), safety_manifest=str(lineage))
        cfg['output_root'] = str(run / 'evaluation')
        cfg['harmbench'].pop('distributed_gcg_import', None)
        evaluation = run / 'evaluation.yaml'
        evaluation.write_text(yaml.safe_dump(cfg, sort_keys=False))
        code = execute([str(ROOT / '.venv-eval/bin/python'), '-u', '-m', 'fineval_pipeline',
            '--config', str(evaluation), 'run', '--models', 'M2', '--tasks', 'fineval'],
            run / 'fineval.log', 'evaluating_FinEval', env)
        if code:
            publish(state='fineval_failed', returncode=code)
            return
        score = json.loads((run / 'evaluation/M2/fineval/summary.json').read_text())
        if score['examples'] != 1151:
            raise ValueError('FinEval capability gate coverage changed')
        correct = round(score['micro_accuracy'] * score['examples'])
        gate = {'correct': correct, 'total': 1151, 'financial_reference_correct': 837,
                'minimum_correct': 796, 'pass': correct >= 796,
                'scope': 'existing Academic val option-logits protocol, not official unlabeled test'}
        write_json(run / 'financial_gate.json', gate)
        publish(state=args.phase_name + '_ready_for_attack_refresh' if gate['pass'] else 'financial_gate_failed',
                checkpoint=str(checkpoint), financial_gate=gate)
    finally:
        lease.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        for i, value in enumerate(sys.argv):
            if value == '--run' and i + 1 < len(sys.argv):
                root = Path(sys.argv[i+1])
                write_json(root / 'supervisor_error.json', {'type': type(error).__name__, 'reason': str(error)})
                path = root / 'status.json'
                state = json.loads(path.read_text()) if path.exists() else {}
                state.update(state='supervisor_failed', error_type=type(error).__name__, reason=str(error))
                write_json(path, state)
        raise
