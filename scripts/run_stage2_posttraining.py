"""Freeze the accepted corpus and run checkpointed SFT, DPO, then GRPO."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import write_json
from scripts.validate_stage2_delivery import native_messages


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def freeze_data(bundle, run):
    scope = json.loads((bundle / 'manifest.json').read_text()).get('training_scope')
    if scope != 'stage2_safety_only':
        raise ValueError('Stage 2 requires the merged safety-only dataset')
    source = bundle / 'validated'
    validation = json.loads((source / 'validation.json').read_text())
    if validation['accepted_records'] < 15000:
        raise ValueError('Full generation must be accepted before the full experiment chain')
    destination = run / 'data'
    destination.mkdir(exist_ok=True)
    files = {}
    for split in ('train', 'validation'):
        sft_rows = rows(source / f'sft_text_{split}.jsonl') + rows(source / f'sft_tool_{split}.jsonl')
        def sft_format(row):
            result = {'messages': native_messages(row['messages'])}
            if row.get('tools'):
                result['tools'] = row['tools']
            return result
        sft = [sft_format(row) for row in sft_rows if row['tokens'] <= 4096]
        long_sft = [sft_format(row) for row in sft_rows if row['tokens'] > 4096]
        dpo = rows(source / f'dpo_{split}.swift.jsonl')
        grpo = rows(source / f'grpo_{split}.jsonl')
        for kind, records in [('sft', sft), ('sft_long', long_sft), ('dpo', dpo),
                              ('grpo', [{'messages': row['prompt'], 'solution': row['id']} for row in grpo])]:
            path = destination / f'{kind}_{split}.jsonl'
            content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records)
            path.write_text(content)
            files[path.name] = {'rows': len(records), 'sha256': hashlib.sha256(content.encode()).hexdigest()}
        # Reward-side copy remains separate from the public GRPO dataset.
        private = destination / f'grpo_{split}.private.jsonl'
        private.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in grpo))
        files[private.name] = {'rows': len(grpo), 'sha256': hashlib.sha256(private.read_bytes()).hexdigest()}
    write_json(run / 'data_manifest.json', {'source': str(bundle), 'training_scope': scope, 'files': files,
        'family_validation': validation, 'private_truth_in_policy_prompt': False})


def configuration(stage, parent, run, world, precision='nf4'):
    algorithm = 'sft' if stage in ('sft', 'sft_long') else stage
    options = {
        'model': str(ROOT / 'models/Qwen3.5-9B'), 'model_type': 'qwen3_5', 'use_hf': True,
        'adapters': [str(parent)], 'load_args': False, 'tuner_type': 'lora',
        'torch_dtype': 'bfloat16', 'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05,
        'target_modules': ['all-linear'], 'freeze_vit': True, 'freeze_aligner': True,
        'gradient_checkpointing': True, 'gradient_checkpointing_kwargs': {'use_reentrant': False},
        'dataset': [str(run / 'data' / f'{stage}_train.jsonl')],
        'val_dataset': [str(run / 'data' / f'{stage}_validation.jsonl')],
        'split_dataset_ratio': 0, 'num_train_epochs': 1, 'max_length': 4096,
        'per_device_train_batch_size': 1, 'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 8 if world == 3 else 16,
        'learning_rate': {'sft': 5e-6, 'sft_long': 2e-6, 'dpo': 2e-6, 'grpo': 1e-6}[stage],
        'warmup_ratio': .05, 'lr_scheduler_type': 'cosine', 'optim': 'adamw_torch_fused',
        'attn_impl': 'sdpa', 'add_non_thinking_prefix': True, 'enable_thinking': False,
        'dataset_num_proc': 2, 'dataloader_num_workers': 0,
        'group_by_length': algorithm == 'sft',
        'eval_strategy': 'epoch', 'save_strategy': 'steps', 'save_steps': 100, 'save_total_limit': 2,
        'logging_steps': 10, 'report_to': ['tensorboard'],
        'external_plugins': [str(ROOT / 'scripts/stage2_audit.py')], 'callbacks': ['stage2_audit'],
        'output_dir': str(run / stage), 'seed': 42, 'data_seed': 42,
        'ddp_find_unused_parameters': False,
    }
    if precision == 'nf4':
        options.update(quant_method='bnb', quant_bits=4, bnb_4bit_quant_type='nf4',
                       bnb_4bit_compute_dtype='bfloat16', bnb_4bit_use_double_quant=True)
    if world == 1 and precision == 'bf16':
        options.update(per_device_train_batch_size={'sft': 6, 'sft_long': 1, 'dpo': 2, 'grpo': 4}[stage],
                       gradient_accumulation_steps={'sft': 4, 'sft_long': 4, 'dpo': 8, 'grpo': 4}[stage])
    if stage == 'sft_long':
        options['max_length'] = 16384
    if algorithm == 'sft':
        options.update(use_liger_kernel=True, liger_kernel_config={
            'rope': False, 'rms_norm': False, 'swiglu': False,
            'cross_entropy': False, 'fused_linear_cross_entropy': True})
    if algorithm != 'sft':
        options.update(rlhf_type=stage, ref_adapters=[str(parent)])
    if stage == 'dpo':
        options.update(beta=.1, rpo_alpha=.2)
    if stage == 'grpo':
        options.update(beta=.02, reward_funcs=['financial_security_verified'],
            num_generations=4, num_generations_eval=4, per_device_eval_batch_size=4,
            generation_batch_size=16 if world == 1 and precision == 'bf16' else 4 * world,
            max_completion_length=512,
            use_vllm=False, log_completions=True)
        options['external_plugins'].append(str(ROOT / 'scripts/stage2_reward_plugin.py'))
    return {'algorithm': algorithm, 'stage': stage, 'parent_checkpoint': str(parent), 'arguments': options}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--precision', choices=['bf16', 'nf4'], default='nf4')
    parser.add_argument('--parent-checkpoint', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.precision != 'nf4':
        raise ValueError('Current experiment protocol requires NF4 training and evaluation')
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    run_lock = (run / '.runner.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'reports/latest_stage2_run.txt').write_text(str(run) + '\n')
    if not (run / 'data_manifest.json').exists():
        freeze_data(args.bundle.resolve(), run)
    data_manifest = json.loads((run / 'data_manifest.json').read_text())
    if data_manifest.get('training_scope') != 'stage2_safety_only':
        raise ValueError('Prepared data predates the safety-only scope; use a new run directory')
    parent = args.parent_checkpoint
    if parent is None:
        first_stage = Path((ROOT / 'reports/latest_merged_full_run.txt').read_text().strip())
        parent = first_stage / 'training/v0-20261106-100608/checkpoint-22802'
    world = len(args.gpus.split(','))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', WANDB_MODE='disabled',
        NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        STAGE2_REWARD_TASKS=os.pathsep.join(str(run / 'data' / f'grpo_{split}.private.jsonl')
                                          for split in ('train', 'validation')))
    status_path = run / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'stages': {}, 'gpu_ids': args.gpus}
    for stage in ('sft', 'sft_long', 'dpo', 'grpo'):
        if stage == 'sft_long' and not data_manifest['files']['sft_long_train.jsonl']['rows']:
            continue
        previous = status['stages'].get(stage, {})
        if previous.get('status') == 'completed':
            parent = Path(previous['checkpoint'])
            continue
        capacity_path = run / 'capacity_plan.json'
        capacity = json.loads(capacity_path.read_text()) if capacity_path.exists() else {}
        if world == 1 and stage not in capacity and not args.prepare_only:
            status['stages'][stage] = {'status': 'profiling', 'parent': str(parent)}
            write_json(status_path, status)
            with (run / (stage + '_capacity.log')).open('a') as log:
                result = subprocess.run([str(ROOT / '.venv-eval/bin/python'), str(ROOT / 'scripts/tune_stage2_capacity.py'),
                    '--run', str(run), '--stage', stage, '--parent', str(parent), '--gpu', args.gpus],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                status['stages'][stage].update(status='failed', reason='capacity_probe_failed')
                write_json(status_path, status)
                raise RuntimeError('Capacity probe failed; inspect ' + str(run / (stage + '_capacity.log')))
            capacity = json.loads(capacity_path.read_text())
        config = configuration(stage, parent, run, world, args.precision)
        if previous.get('resume_checkpoint'):
            config['arguments']['resume_from_checkpoint'] = previous['resume_checkpoint']
        if stage in capacity:
            config['arguments'].update(per_device_train_batch_size=capacity[stage]['micro_batch'],
                gradient_accumulation_steps=capacity[stage]['gradient_accumulation'])
            if stage == 'grpo':
                config['arguments']['generation_batch_size'] = capacity[stage].get(
                    'generation_batch_size', 4 * capacity[stage]['micro_batch'])
        if not data_manifest['files'][stage + '_validation.jsonl']['rows']:
            config['arguments'].update(val_dataset=[], eval_strategy='no')
        config_path = run / (stage + '_config.json')
        write_json(config_path, config)
        if args.prepare_only:
            print(config_path)
            break
        python = str(ROOT / '.venv-train-fast/bin/python')
        command = [python, str(ROOT / 'scripts/stage2_swift_entry.py'), str(config_path)]
        if world > 1:
            command = [python, '-m', 'torch.distributed.run', '--nproc_per_node', str(world),
                       '--master_port', '29691', str(ROOT / 'scripts/stage2_swift_entry.py'), str(config_path)]
        entry = {'status': 'running', 'parent': str(parent), 'started': datetime.datetime.now().isoformat(),
                 'config': str(config_path), 'command': command}
        status['stages'][stage] = entry
        write_json(status_path, status)
        with (run / (stage + '.log')).open('a') as log:
            env.update(STAGE2_RUN=str(run), STAGE2_STAGE=stage, STAGE2_PARENT=str(parent))
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry['pid'] = child.pid
            write_json(status_path, status)
            code = child.wait()
        if code:
            entry.update(status='failed', returncode=code)
            write_json(status_path, status)
            raise RuntimeError(f'{stage} failed; inspect {run / (stage + ".log")}')
        states = list((run / stage).glob('**/checkpoint-*/trainer_state.json'))
        checkpoint = max(states, key=lambda path: json.loads(path.read_text())['global_step']).parent
        if not (checkpoint / 'adapter_model.safetensors').exists():
            raise RuntimeError('Missing stage checkpoint adapter')
        entry.update(status='completed', checkpoint=str(checkpoint), ended=datetime.datetime.now().isoformat())
        write_json(status_path, status)
        parent = checkpoint


if __name__ == '__main__':
    main()
