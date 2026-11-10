"""Freeze native Qwen training files with family isolation and token accounting."""
import argparse
import collections
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import digest, fingerprint, rows, write_json
from fineval_pipeline.hardening_data import ProtectedIndex, protected_entries, WRAPPERS, training_attack_sources
from scripts.validate_stage2_delivery import native_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-training-attacks', action='store_true')
    parser.add_argument('--focus-training-attacks', action='store_true')
    args = parser.parse_args()
    run = args.output.absolute()
    if (run / 'data_manifest.json').exists():
        print('FROZEN_DATA_EXISTS', flush=True)
        return
    data = ROOT / 'data/security_hardening_20261109'
    seed_manifest = json.loads((data / 'seed_manifest.json').read_text())
    if digest(data / 'seed_records.jsonl') != seed_manifest['seed_sha256']:
        raise ValueError('Seed curriculum changed')
    seeds = rows(data / 'seed_records.jsonl')
    seed_by_id = {r['id']: r for r in seeds}
    records = []
    attack_inputs = []
    for row in seeds:
        records.append(row)
        if row['source'].startswith('PKU'):
            kind, prefix, suffix = WRAPPERS[int(row['family_id'][:8], 16) % len(WRAPPERS)]
            records.append({**row, 'id': row['id'] + ':' + kind, 'prompt': prefix + row['prompt'] + suffix,
                            'variant': 'static_' + kind})
    if (data / 'teacher_candidates.jsonl').exists():
        records += rows(data / 'teacher_candidates.jsonl')
    if args.include_training_attacks:
        folder = ROOT / 'outputs/security_hardening_20261109_v1/training_attacks'
        for path, method in training_attack_sources(folder):
            payload = json.loads(path.read_text())
            attack_inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                'method': method, 'source_ids': sorted(payload), 'cases': sum(len(x) for x in payload.values())})
            for source_id, prompts in payload.items():
                source = seed_by_id[source_id]
                if source['split'] != 'train':
                    raise ValueError('Attempted to train on a development attack')
                for i, prompt in enumerate(prompts):
                    records.append({**source, 'id': source_id + ':' + method + ':' + str(i) + ':' + digest(path)[:12],
                        'prompt': prompt, 'variant': method + '_generated',
                        'attack_case_sha256': digest(path)})
    if args.focus_training_attacks:
        if not args.include_training_attacks:
            raise ValueError('Attack-focused training requires actual training attacks')
        attack_status = ROOT / 'outputs/security_hardening_20261109_v1/training_attacks/status.json'
        if not attack_status.exists() or json.loads(attack_status.read_text()).get('state') != 'completed':
            raise ValueError('Attack-focused data waits for the live training-attack campaign to finish')
        attacks = [r for r in records if r['variant'] in ('GCG_generated', 'PAIR_generated')]
        unique_attacks = {}
        for row in attacks:
            unique_attacks.setdefault((row['family_id'], row['prompt']), row)
        attacks = list(unique_attacks.values())
        if len(attacks) < 48 or not any(r['variant'] == 'PAIR_generated' for r in attacks):
            raise ValueError('Attack-focused stage is waiting for sufficient GCG and PAIR cases')
        rng = random.Random(20261109)
        replay = []
        for task, count in [('benign_boundary', 512), ('privacy_behavior', 128),
                            ('crypto_detection', 64), ('agent_tool_use', 64)]:
            pool = [r for r in seeds if r['split'] == 'train' and r['task_type'] == task]
            rng.shuffle(pool)
            replay += pool[:count]
        teacher = [r for r in records if r['variant'] == 'teacher_adversarial']
        development = [r for r in records if r['split'] == 'development']
        repeated = [{**r, 'id': r['id'] + ':repeat' + str(i), 'oversample_repeat': i}
                    for r in attacks for i in range(8)]
        records = development + replay + teacher + repeated
    # Newly generated text is screened again, independently of seed acceptance.
    guard = ProtectedIndex(protected_entries())
    generated = [r for r in records if r['variant'] != 'original']
    rejected = []
    accepted_generated = []
    for start in range(0, len(generated), 128):
        batch = generated[start:start+128]
        prompt_checks = guard.screen([r['prompt'] for r in batch])
        answer_checks = guard.screen([r['answer'] for r in batch], threshold=.70)
        for r, p, a in zip(batch, prompt_checks, answer_checks):
            if p['excluded'] or a['excluded']:
                rejected.append({'id': r['id'], 'reason': 'protected_similarity', 'prompt': p, 'answer': a})
            else:
                accepted_generated.append(r)
    records = [r for r in records if r['variant'] == 'original'] + accepted_generated
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3.5-9B', local_files_only=True)
    prepared = {'train': [], 'development': []}
    seen = {}
    for row in records:
        if row['variant'] == 'original' and row.get('original_messages'):
            messages = native_messages(row['original_messages'])
        else:
            messages = [{'role': 'user', 'content': row['prompt']},
                        {'role': 'assistant', 'content': row['answer']}]
        native = {'messages': messages}
        if row.get('tools') and row['variant'] == 'original':
            native['tools'] = row['tools']
        length = len(tokenizer.apply_chat_template(messages, tools=native.get('tools') or None,
            return_dict=False, enable_thinking=False))
        if length > 4050:
            rejected.append({'id': row['id'], 'reason': 'length_buffer_for_native_training_prefix', 'tokens': length})
            continue
        key = fingerprint(native)
        if key in seen:
            if seen[key] != row['split']:
                raise ValueError('Identical messages crossed training/development splits')
            if not args.focus_training_attacks or 'oversample_repeat' not in row:
                continue
        seen[key] = row['split']
        prepared[row['split']].append((native, {k: row.get(k) for k in
            ['id', 'family_id', 'split', 'source', 'task_type', 'variant', 'teacher_review',
             'attack_case_sha256', 'oversample_repeat']}, length))
    families = {s: {r[1]['family_id'] for r in v} for s, v in prepared.items()}
    if families['train'] & families['development']:
        raise ValueError('Curriculum family leakage')
    minimum = 1000 if args.focus_training_attacks else 5000
    if len(prepared['train']) < minimum or len(prepared['development']) < 100:
        raise ValueError('Insufficient screened data for this experiment')
    folder = run / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    variants = {}
    longest = None
    for split, label in [('train', 'train'), ('development', 'validation')]:
        items = prepared[split]
        path = folder / ('sft_' + label + '.jsonl')
        path.write_text(''.join(json.dumps(x[0], ensure_ascii=False) + '\n' for x in items))
        files[path.name] = {'rows': len(items), 'sha256': digest(path)}
        provenance = folder / ('sft_' + label + '.records.jsonl')
        provenance.write_text(''.join(json.dumps({**x[1], 'tokens': x[2]}, ensure_ascii=False) + '\n' for x in items))
        files[provenance.name] = {'rows': len(items), 'sha256': digest(provenance)}
        variants[split] = dict(collections.Counter(x[1]['variant'] for x in items))
        if split == 'train':
            longest = max(items, key=lambda x: x[2])
    stress = folder / 'stress.jsonl'
    stress.write_text((json.dumps(longest[0], ensure_ascii=False) + '\n') * 64)
    files[stress.name] = {'rows': 64, 'sha256': digest(stress)}
    write_json(run / 'data_rejections.json', rejected)
    write_json(run / 'data_manifest.json', {'training_scope': 'stage2_safety_adversarial_only',
        'files': files, 'variants': variants, 'maximum_train_tokens': longest[2],
        'seed_manifest_sha256': digest(data / 'seed_manifest.json'),
        'family_intersection': 0, 'official_test_used_for_training': False,
        'lexical_guard_scope': seed_manifest['guard_scope'],
        'includes_training_attacks': args.include_training_attacks,
        'training_attack_inputs': attack_inputs,
        'curriculum_builder_sha256': digest(__file__),
        'curriculum_util_sha256': digest(ROOT / 'fineval_pipeline/hardening_data.py'),
        'attack_focused': args.focus_training_attacks,
        'training_attack_oversampling': 8 if args.focus_training_attacks else 1,
        'teacher_scope': 'automated independent-context review; not expert certification'})
    print('CURRICULUM_FROZEN', json.dumps({'rows': {s: len(v) for s, v in prepared.items()},
        'variants': variants, 'longest': longest[2]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
