"""Tokenize, deduplicate and split generated records by shared task families."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import write_json
from scripts.validate_pilot_data import SimilarityIndex, normalized


def native_messages(messages):
    messages = json.loads(json.dumps(messages))
    for message in messages:
        for call in message.get('tool_calls', []):
            arguments = call['function']['arguments']
            if isinstance(arguments, str):
                call['function']['arguments'] = json.loads(arguments)
    return messages


def encoded(tokenizer, row):
    if row['record_type'] == 'dpo':
        return max(len(tokenizer.apply_chat_template(row['prompt'] + row[branch],
                       return_dict=False, enable_thinking=False)) for branch in ('chosen', 'rejected'))
    messages = native_messages(row.get('messages', row.get('prompt')))
    if row['record_type'] == 'grpo':
        messages = messages + [{'role': 'assistant', 'content': row['reference_completion']}]
    return len(tokenizer.apply_chat_template(messages, tools=row.get('tools') or None,
                                             return_dict=False, enable_thinking=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3.5-9B', local_files_only=True)
    records = []
    for name in ('sft_text', 'sft_tools', 'dpo_pairs', 'grpo_tasks'):
        for line in (args.bundle / (name + '.jsonl')).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    families = {row['family_id']: row['family_id'] for row in records}

    def root(family):
        while families[family] != family:
            families[family] = families[families[family]]
            family = families[family]
        return family

    index = SimilarityIndex()
    seen_text, accepted, rejections = set(), [], []
    for row in records:
        messages = row.get('messages', row.get('prompt'))
        prompt = ' '.join(m['content'] for m in messages if m['role'] == 'user')
        text = normalized(prompt, True)
        matches = index.search(text, protected=True)
        for match in matches:
            families[root(row['family_id'])] = root(match['reference'])
        index.add(row['family_id'], text)
        identity_fields = {'record_type': row['record_type'], 'messages': messages,
                           'chosen': row.get('chosen'), 'rejected': row.get('rejected'),
                           'reference_completion': row.get('reference_completion')}
        fingerprint = hashlib.sha256(json.dumps(identity_fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if fingerprint in seen_text:
            rejections.append({'id': row['id'], 'reason': 'exact_duplicate'})
            continue
        seen_text.add(fingerprint)
        try:
            tokens = encoded(tokenizer, row)
        except Exception as exc:
            rejections.append({'id': row['id'], 'reason': 'native_encoding', 'detail': str(exc)[:200]})
            continue
        limit = 16384 if row['record_type'] == 'sft_tool' else 4096
        if tokens > limit:
            rejections.append({'id': row['id'], 'reason': 'over_length', 'tokens': tokens})
            continue
        row['tokens'] = tokens
        accepted.append(row)
    output = args.bundle / 'validated'
    output.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for row in accepted:
        row['family_id_before_clustering'] = row['family_id']
        row['family_id'] = root(row['family_id'])
        grouped[row['family_id']].append(row)
    buckets = defaultdict(list)
    for family, rows in grouped.items():
        fraction = int(hashlib.sha256(('42:' + family).encode()).hexdigest()[:8], 16) / 2**32
        split = 'train' if fraction < .9 else 'validation' if fraction < .95 else 'test'
        fixed = {row['fixed_split'] for row in rows if row.get('fixed_split')}
        if fixed:
            split = next(name for name in ('test', 'validation', 'train') if name in fixed)
        for row in rows:
            row['split'] = split
            buckets[(row['record_type'], split)].append(row)
    for kind in ('sft_text', 'sft_tool', 'dpo', 'grpo'):
        for split in ('train', 'validation', 'test'):
            path = output / f'{kind}_{split}.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                    for row in buckets[(kind, split)]))
    for split in ('train', 'validation', 'test'):
        sft = buckets[('sft_text', split)] + buckets[('sft_tool', split)]
        (output / f'sft_{split}.swift.jsonl').write_text(''.join(json.dumps(
            dict({'messages': native_messages(row['messages'])},
                 **({'tools': row['tools']} if row.get('tools') else {})), ensure_ascii=False) + '\n' for row in sft))
        (output / f'dpo_{split}.swift.jsonl').write_text(''.join(json.dumps(
            {'messages': row['prompt'] + row['chosen'], 'rejected_response': row['rejected'][-1]['content']},
            ensure_ascii=False) + '\n' for row in buckets[('dpo', split)]))
        (output / f'grpo_{split}.public.jsonl').write_text(''.join(json.dumps(
            {key: row[key] for key in ('id', 'family_id', 'task_type', 'prompt', 'tools') if key in row},
            ensure_ascii=False) + '\n' for row in buckets[('grpo', split)]))
    validation = {
        'status': 'native_encoding_and_family_isolation_passed_pending_expert_sampling',
        'input_records': len(records), 'accepted_records': len(accepted), 'families': len(grouped),
        'counts': {kind + ':' + split: len(rows) for (kind, split), rows in buckets.items()},
        'tokens': {'max': max((row['tokens'] for row in accepted), default=0),
                   'mean': sum(row['tokens'] for row in accepted) / max(1, len(accepted))},
        'rejections': rejections,
        'family_isolation': {split: sorted({row['family_id'] for row in accepted if row['split'] == split})
                             for split in ('train', 'validation', 'test')},
        'limitations': ['teacher-reviewed crypto truth requires expert sampling',
                        'repair runtime tests and full tool-decision environments pending',
                        'near-duplicate and source-family coverage limited by available metadata']}
    write_json(output / 'validation.json', validation)
    manifest_path = args.bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['validation'] = {'path': 'validated/validation.json', 'accepted_records': len(accepted),
                              'families': len(grouped), 'counts': validation['counts']}
    manifest['files'] = {str(path.relative_to(args.bundle)): {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
        for path in args.bundle.rglob('*') if path.is_file() and path.name not in ('manifest.json', 'status.json')}
    write_json(manifest_path, manifest)
    write_json(ROOT / 'reports/stage2_deepseek_generation_latest.json', {'output': str(args.bundle.resolve()), **manifest})
    print(json.dumps({'accepted': len(accepted), 'families': len(grouped),
                      'counts': {kind + ':' + split: len(rows) for (kind, split), rows in buckets.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
