"""Freeze human safe/unsafe preferences plus independently reviewed train attacks."""
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import digest, fingerprint, rows, write_json
from fineval_pipeline.hardening_data import ProtectedIndex, protected_entries, content, user_prompt
from scripts.validate_stage2_delivery import native_messages


def main():
    root = ROOT / 'outputs/security_hardening_20261109_v1'
    out = root / 'adversarial_dpo_v1'
    out.mkdir(exist_ok=True)
    if (out / 'data_manifest.json').exists():
        print('FROZEN_PREFERENCE_DATA_EXISTS', flush=True)
        return
    harvest = root / 'preference_refresh_dpo'
    state = json.loads((harvest / 'status.json').read_text())
    if state['state'] not in ('completed', 'review_budget_limited'):
        raise ValueError('Wait for reviewed training-only model counterexamples')
    seed_path = ROOT / 'data/security_hardening_20261109/seed_records.jsonl'
    seeds = rows(seed_path)
    records = []
    for row in seeds:
        if row['split'] != 'train' or not row['source'].startswith('PKU'):
            continue
        records.append({'id': row['id'] + ':human_preference', 'family_id': row['family_id'],
            'source': 'PKU_human_safe_unsafe', 'prompt': [{'role': 'user', 'content': row['prompt']}],
            'chosen': row['answer'], 'rejected': row['negative_answer'], 'split': 'train',
            'task_type': 'human_safe_response_preference'})
    reviewed = rows(harvest / 'accepted_pairs.jsonl')
    for row in reviewed:
        review = row['review']
        if not (review['candidate_label'] == 'unsafe' and review['reference_safe']
                and review['reference_appropriate'] and review['preference_clear']):
            raise ValueError('An unreviewed or unclear adversarial preference was admitted')
        for repeat in range(8):
            records.append({'id': row['id'] + ':repeat' + str(repeat), 'family_id': row['family_id'],
                'source': 'actual_DPO_adversarial_counterexample',
                'prompt': [{'role': 'user', 'content': row['prompt']}], 'chosen': row['chosen'],
                'rejected': row['rejected'], 'split': 'train', 'task_type': 'adversarial_preference',
                'oversample_repeat': repeat, 'review': review, 'case_sha256': row['case_sha256']})
    source = ROOT / 'data-distill/stage2_safety_merged_20261107_v1/validated'
    old = rows(source / 'dpo_train.jsonl')
    rng = random.Random(20261109)
    for task, count in [('benign_boundary', 512), ('privacy_behavior', 128), ('crypto_detection', 128)]:
        pool = [row for row in old if row['task_type'] == task]
        rng.shuffle(pool)
        if len(pool) < count:
            raise ValueError('Insufficient protected-utility replay for ' + task)
        for row in pool[:count]:
            records.append({'id': 'replay:' + row['id'], 'family_id': row['family_id'],
                'source': 'existing_DPO_training_replay', 'prompt': native_messages(row['prompt']),
                'chosen': content(row['chosen']), 'rejected': content(row['rejected']), 'split': 'train',
                'task_type': task})
    for row in rows(source / 'dpo_validation.jsonl'):
        records.append({'id': 'validation:' + row['id'], 'family_id': row['family_id'],
            'source': 'fixed_old_DPO_validation', 'prompt': native_messages(row['prompt']),
            'chosen': content(row['chosen']), 'rejected': content(row['rejected']),
            'split': 'validation', 'task_type': row['task_type']})
    guard = ProtectedIndex(protected_entries())
    screened, rejected = [], []
    training = [r for r in records if r['split'] == 'train']
    for start in range(0, len(training), 128):
        batch = training[start:start + 128]
        prompt_matches = guard.screen([user_prompt({'messages': r['prompt']}) for r in batch])
        chosen_matches = guard.screen([r['chosen'] for r in batch], threshold=.70)
        rejected_matches = guard.screen([r['rejected'] for r in batch], threshold=.70)
        for row, p, c, r in zip(batch, prompt_matches, chosen_matches, rejected_matches):
            if p['excluded'] or c['excluded'] or r['excluded']:
                rejected.append({'id': row['id'], 'reason': 'protected_similarity',
                    'prompt': p, 'chosen': c, 'rejected': r})
            else:
                screened.append(row)
    records = screened + [r for r in records if r['split'] == 'validation']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3.5-9B', local_files_only=True)
    prepared = {'train': [], 'validation': []}
    seen = {}
    for row in records:
        if not row['chosen'].strip() or not row['rejected'].strip() or row['chosen'].strip() == row['rejected'].strip():
            rejected.append({'id': row['id'], 'reason': 'empty_or_identical_preference'})
            continue
        messages = row['prompt'] + [{'role': 'assistant', 'content': row['chosen']}]
        other = row['prompt'] + [{'role': 'assistant', 'content': row['rejected']}]
        length = max(len(tokenizer.apply_chat_template(branch, return_dict=False, enable_thinking=False))
                     for branch in [messages, other])
        if length > 4050:
            rejected.append({'id': row['id'], 'reason': 'native_pair_over_length', 'tokens': length})
            continue
        payload = {'messages': messages, 'rejected_response': row['rejected']}
        key = fingerprint(payload)
        if key in seen:
            if seen[key] != row['split']:
                raise ValueError('A preference crossed train and validation splits')
            if 'oversample_repeat' not in row:
                continue
        seen[key] = row['split']
        prepared[row['split']].append((payload, {k: row.get(k) for k in ['id', 'family_id', 'source',
            'task_type', 'oversample_repeat', 'review', 'case_sha256']}, length))
    families = {split: {row[1]['family_id'] for row in data} for split, data in prepared.items()}
    prompts = {split: {fingerprint(row[0]['messages'][:-1]) for row in data} for split, data in prepared.items()}
    if families['train'] & families['validation'] or prompts['train'] & prompts['validation']:
        raise ValueError('New DPO data leaks a validation prompt/family into training')
    if len(prepared['train']) < 2000 or len(prepared['validation']) != 492:
        raise ValueError('Insufficient preference data or changed fixed validation cohort')
    data = out / 'data'
    data.mkdir(exist_ok=True)
    files = {}
    for split, items in prepared.items():
        for name, values in [('dpo_' + split + '.jsonl', [item[0] for item in items]),
                             ('dpo_' + split + '.records.jsonl', [{**item[1], 'tokens': item[2]} for item in items])]:
            path = data / name
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values))
            files[name] = {'rows': len(values), 'sha256': digest(path)}
    longest = max(prepared['train'], key=lambda row: row[2])
    path = data / 'stress.jsonl'
    path.write_text((json.dumps(longest[0], ensure_ascii=False) + '\n') * 64)
    files[path.name] = {'rows': 64, 'sha256': digest(path)}
    write_json(out / 'data_rejections.json', rejected)
    write_json(out / 'data_manifest.json', {'algorithm': 'dpo', 'training_scope': 'stage2_safety_adversarial_only',
        'files': files, 'family_intersection': 0, 'prompt_intersection': 0,
        'maximum_train_tokens': longest[2], 'reviewed_adversarial_candidates': len(reviewed),
        'actual_adversarial_unique_pairs': len({row[1]['id'].split(':repeat')[0] for row in prepared['train']
            if row[1]['source'] == 'actual_DPO_adversarial_counterexample'}),
        'adversarial_repeat_factor': 8, 'teacher_review_scope': 'automated independent-context Flash labels, not expert truth',
        'counts': {split: dict(Counter(row[1]['source'] for row in data)) for split, data in prepared.items()},
        'source_seed_sha256': digest(seed_path), 'reviewed_pairs_sha256': digest(harvest / 'accepted_pairs.jsonl'),
        'original_validation_sha256': digest(source / 'dpo_validation.jsonl'),
        'official_test_used_for_training': False, 'first_stage_financial_SFT_repeated': False,
        'builder_sha256': digest(__file__), 'lexical_guard_scope': 'exact/substring + character TF-IDF, not semantic proof'})
    print('ADVERSARIAL_DPO_DATA_FROZEN', json.dumps({split: len(data) for split, data in prepared.items()}), flush=True)


if __name__ == '__main__':
    main()
