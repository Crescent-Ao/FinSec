"""Prepare or execute isolated safety SFT; test sets are never training sources."""
import json
import subprocess
import sys

from .core import ROOT, digest, fingerprint, model_identity, model_spec, resolve, rows, write_json


def build_plan(cfg, execute=False):
    settings = cfg['safety_training']
    if settings.get('quant_bits') not in (None, 4):
        raise ValueError('Safety quantization currently supports only 4-bit NF4')
    m1 = model_spec(cfg, 'M1')
    output = resolve(settings['output'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new safety output directory; never overwrite M1 or an existing M2')
    train_path, validation_path = resolve(settings['train']), resolve(settings['validation'])
    training, validation = rows(train_path), rows(validation_path)
    if not training or not validation:
        raise ValueError('Safety train and validation must both be nonempty')
    protected = rows(resolve(cfg['privacy']['nonmembers']))
    protected_ids = {r['id'] for r in protected}
    protected_groups = {r['group_id'] for r in protected}
    protected_text = {fingerprint(r['messages']) for r in protected}
    train_ids, train_groups, seen_text = set(), set(), set()
    from scripts.validate_pilot_data import SimilarityIndex, load_fineval, normalized
    fineval, count = load_fineval(resolve(cfg['fineval']['data_dir']).parent)
    if count < 4661:
        raise ValueError('Full FinEval guard is required before safety SFT')
    hb = __import__('csv')
    with resolve(cfg['harmbench']['behaviors']).open() as f:
        protected_behaviors = list(hb.DictReader(f))
    index = SimilarityIndex()
    for r in protected:
        index.add(r['id'], normalized(' '.join(x['content'] for x in r['messages'])))
    for b in protected_behaviors:
        index.add(b['BehaviorID'], normalized(b['Behavior']+' '+b['ContextString']))
    for case in rows(resolve(cfg['agent']['cases'])):
        index.add(case['id'], normalized(case['code']))
    for split, records in [('train', training), ('validation', validation)]:
        ids, groups = set(), set()
        for r in records:
            identity = fingerprint(r['messages'])
            if r['id'] in ids or identity in seen_text:
                raise ValueError('Duplicate or cross-split safety sample')
            if r['id'] in protected_ids or r['group_id'] in protected_groups or identity in protected_text:
                raise ValueError('Privacy nonmember leaked into safety training/validation')
            text = normalized(' '.join(x['content'] for x in r['messages']))
            if fineval.search(text, protected=True) or index.search(text, protected=True):
                raise ValueError('Safety data overlaps protected FinEval/HarmBench/nonmember material')
            if any(x in str(r.get('source', '')).lower() for x in ('fineval', 'harmbench_test', 'crypto_audit/cases')):
                raise ValueError('Evaluation source declared as safety training source')
            ids.add(r['id']); groups.add(r['group_id']); seen_text.add(identity)
        if split=='train':
            train_ids, train_groups = ids, groups
        elif ids & train_ids or groups & train_groups:
            raise ValueError('Safety train/validation families overlap')
    output.mkdir(parents=True)
    for split, records in [('train', training), ('validation', validation)]:
        (output/f'{split}.jsonl').write_text(''.join(json.dumps({'messages': r['messages']}, ensure_ascii=False)+'\n' for r in records))
    command = [str(ROOT/'.venv/bin/swift'), 'sft', '--model', m1['path'], '--model_type', 'qwen3_5',
               '--use_hf', 'true', '--tuner_type', 'lora', '--torch_dtype', 'bfloat16',
               '--dataset', str(output/'train.jsonl'), '--val_dataset', str(output/'validation.jsonl'),
               '--split_dataset_ratio', '0', '--num_train_epochs', str(settings['epochs']),
               '--learning_rate', str(settings['learning_rate']), '--max_length', str(settings.get('max_length', 2048)),
               '--per_device_train_batch_size', str(settings.get('per_device_train_batch_size', 2)),
               '--gradient_accumulation_steps', str(settings.get('gradient_accumulation_steps', 8)),
               '--gradient_checkpointing', 'true', '--freeze_vit', 'true', '--freeze_aligner', 'true',
               '--lora_rank', str(settings.get('lora_rank', 32)), '--lora_alpha', str(settings.get('lora_alpha', 64)), '--target_modules', 'all-linear',
               '--add_non_thinking_prefix', 'true', '--attn_impl', 'sdpa', '--seed', str(cfg['seed']),
               '--eval_strategy', 'epoch', '--save_strategy', 'epoch', '--save_total_limit', '2',
               '--load_best_model_at_end', 'true', '--metric_for_best_model', 'loss', '--greater_is_better', 'false',
               '--report_to', 'none', '--output_dir', str(output/'training')]
    if settings.get('quant_bits') is not None:
        command += ['--quant_method', 'bnb', '--quant_bits', '4', '--bnb_4bit_quant_type', 'nf4',
                    '--bnb_4bit_compute_dtype', 'bfloat16', '--bnb_4bit_use_double_quant', 'true']
    if m1.get('adapter'):
        command += ['--resume_from_checkpoint', m1['adapter'], '--resume_only_model', 'true']
    plan = {'parent': 'M1', 'parent_spec': m1, 'parent_identity': model_identity(m1), 'command': command,
            'train_sha256': digest(train_path), 'validation_sha256': digest(validation_path),
            'train_ids': sorted(train_ids), 'train_groups': sorted(train_groups),
            'privacy_nonmembers_sha256': digest(resolve(cfg['privacy']['nonmembers'])),
            'nonmembers_excluded': True, 'fineval_guard_questions': count,
            'selection': 'safety validation loss only, never test ASR or FinEval',
            'registration': 'Set M2.path to parent base/full model, M2.adapter to best newly saved adapter, record this plan as safety_manifest.'}
    write_json(output/'plan.json', plan)
    if execute:
        with (output/'train.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return plan
