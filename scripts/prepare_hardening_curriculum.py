"""Prepare public train-only safety examples and protected, family-level splits."""
import collections
import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import append, digest, fingerprint, write_json
from fineval_pipeline.hardening_data import (ProtectedIndex, protected_entries,
    cluster_groups, group_id, content, user_prompt, WRAPPERS)


def main():
    output = ROOT / 'data/security_hardening_20261109'
    if (output / 'seed_manifest.json').exists():
        print('FROZEN_SEED_CURRICULUM_EXISTS', flush=True)
        return
    raw = output / 'raw/pku_alpaca3_train.jsonl'
    source_manifest = json.loads((raw.parent / 'source_manifest.json').read_text())
    if digest(raw) != source_manifest['sha256']:
        raise ValueError('Public training source changed')
    candidates = []
    for index, line in enumerate(raw.open()):
        row = json.loads(line)
        safe = [row['is_response_0_safe'], row['is_response_1_safe']]
        if sum(safe) != 1:
            continue
        chosen = safe.index(True)
        answer = row['response_' + str(chosen)]
        if not 40 <= len(answer) <= 4000 or not 20 <= len(row['prompt']) <= 2500:
            continue
        candidates.append({'id': 'pku_' + str(index), 'prompt': row['prompt'], 'answer': answer,
            'negative_answer': row['response_' + str(1-chosen)], 'source': 'PKU-SafeRLHF:train',
            'task_type': 'safe_response_preference', 'variant': 'original',
            'note': 'Response safety annotations do not by themselves label prompt harmfulness.'})
    protected = protected_entries()
    print('PROTECTED_ENTRIES', len(protected), 'PUBLIC_CANDIDATES', len(candidates), flush=True)
    guard = ProtectedIndex(protected)
    accepted = []
    excluded = []
    for start in range(0, len(candidates), 128):
        batch = candidates[start:start+128]
        # Screen both user text and the supervised response against held-out material.
        prompts = guard.screen([r['prompt'] for r in batch])
        answers = guard.screen([r['answer'] for r in batch], threshold=.70)
        for row, p, a in zip(batch, prompts, answers):
            if p['excluded'] or a['excluded']:
                excluded.append({'id': row['id'], 'prompt_match': p, 'answer_match': a})
            else:
                accepted.append(row)
    old = ROOT / 'data-distill/stage2_safety_merged_20261107_v1/validated'
    for path in [old / 'sft_text_train.jsonl', old / 'sft_tool_train.jsonl']:
        for line in path.open():
            row = json.loads(line)
            if row['tokens'] > 4096:
                continue
            # Existing samples retain tools/messages; the teacher receives only plain prompts.
            accepted.append({'id': 'existing_' + row['id'], 'prompt': user_prompt(row),
                'answer': content(row['messages'][-1]), 'source': 'existing_stage2_train',
                'task_type': row['task_type'], 'variant': 'original',
                'original_messages': row['messages'], 'tools': row.get('tools')})
    # Re-screen the combined set, including previously accepted training sources.
    final = []
    for start in range(0, len(accepted), 128):
        batch = accepted[start:start+128]
        answer_checks = guard.screen([r['answer'] for r in batch], threshold=.70)
        for row, check, answer_check in zip(batch, guard.screen([r['prompt'] for r in batch]), answer_checks):
            if check['excluded'] or answer_check['excluded']:
                excluded.append({'id': row['id'], 'prompt_match': check, 'answer_match': answer_check})
            else:
                final.append(row)
    print('CLUSTERING', len(final), flush=True)
    cluster_groups(final)
    # Existing training examples were seen by parent checkpoints and cannot be
    # used to select a new model on a supposedly unseen development split.
    previously_seen = {r['family_id'] for r in final if r['source'] == 'existing_stage2_train'}
    for row in final:
        if row['family_id'] in previously_seen:
            row['split'] = 'train'
    # No duplicate canonical prompt may cross splits; keep its first supervised target.
    unique = {}
    for row in final:
        key = fingerprint(row['original_messages']) if row.get('original_messages') else group_id(row['prompt'])
        unique.setdefault(key, row)
    final = list(unique.values())
    seed = output / 'seed_records.jsonl'
    seed.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in final))
    rng = random.Random(20261109)
    train_public = [r for r in final if r['split'] == 'train' and r['source'].startswith('PKU')]
    rng.shuffle(train_public)
    teacher = train_public[:256]
    benign = [r for r in final if r['split'] == 'train' and r['task_type'] == 'benign_boundary']
    rng.shuffle(benign)
    teacher += benign[:128]
    (output / 'teacher_seeds.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in teacher))
    attack_rows = train_public[:48]
    fields = ['Behavior', 'FunctionalCategory', 'SemanticCategory', 'Tags', 'ContextString', 'BehaviorID']
    for method, count in [('gcg', 48), ('pair', 24)]:
        with (output / (method + '_train_behaviors.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in attack_rows[:count]:
                writer.writerow({'Behavior': row['prompt'], 'FunctionalCategory': 'standard',
                    'SemanticCategory': 'independent_train_only', 'Tags': 'PKU-train',
                    'ContextString': '', 'BehaviorID': row['id']})
    # Target strings come only from the source training negative response, never official targets.
    targets = {r['id']: r['negative_answer'][:180] for r in attack_rows}
    write_json(output / 'train_attack_targets.json', targets)
    write_json(output / 'exclusions.json', excluded)
    counts = collections.Counter(r['split'] + ':' + r['source'] for r in final)
    manifest = {'scope': 'stage2_safety_adversarial_only', 'source': source_manifest,
        'seed_records': len(final), 'counts': dict(counts), 'protected_entries': len(protected),
        'excluded': len(excluded), 'seed_sha256': digest(seed),
        'split_policy': 'near-duplicate prompt clusters; all attack variants inherit the source family split',
        'teacher_seeds': len(teacher), 'GCG_train_intents': 48, 'PAIR_train_intents': 24,
        'test_evaluation_material_in_training': False,
        'guard_scope': 'exact/substring and character TF-IDF exclusion; not a semantic non-overlap proof',
        'lexical_prompt_threshold': .55, 'lexical_answer_threshold': .70}
    write_json(output / 'seed_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
