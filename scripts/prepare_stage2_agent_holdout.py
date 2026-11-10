"""Freeze file-level audit fixtures from already isolated stage-2 test families."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import digest, rows, write_json


def main():
    bundle = ROOT / 'data-distill/stage2_safety_merged_20261107_v1/validated'
    old = ROOT / 'outputs/merged_full_ddp3_20261106T015850Z/cpu_preparation/agent'
    out = ROOT / 'evaluation_data/stage2_agent_holdout_20261107_v1'
    out.mkdir(exist_ok=True)
    forbidden = set()
    for kind in ('sft_text', 'sft_tool', 'dpo', 'grpo'):
        for split in ('train', 'validation'):
            forbidden.update(r['family_id'] for r in rows(bundle / f'{kind}_{split}.jsonl'))
    knowledge = json.loads((old / 'knowledge.json').read_text())
    allowed = {r['id'] for r in knowledge}
    cases = rows(old / 'agent_heldout.jsonl')
    originals = len(cases)
    generated = rows(bundle / 'grpo_test.jsonl')
    for record in generated:
        assert record['family_id'] not in forbidden
        truth = record['private_ground_truth']
        for file in truth['files']:
            ast.parse(file['code'])
            expected = [f for f in truth['findings'] if f['file'] == file['path']]
            lines = file['code'].splitlines()
            for finding in expected:
                assert finding['rule_id'] in allowed
                assert 1 <= finding['line'] <= len(lines)
                assert lines[finding['line']-1].strip() == finding['evidence'].strip()
            cases.append({'id': record['id'] + ':' + file['path'], 'family': record['family_id'],
                'file': file['path'], 'code': file['code'], 'expected': expected,
                'source_record': record['id'], 'source_split': 'test',
                'label_scope': 'synthetic teacher labels; static evidence checked; expert review pending'})
    seen, unique, duplicate_ids = {}, [], []
    for case in cases:
        key = hashlib.sha256(case['code'].encode()).hexdigest()
        labels = sorted((f['rule_id'], f['line']) for f in case['expected'])
        if key in seen:
            if seen[key] != labels:
                raise ValueError('Conflicting audit labels for identical code')
            duplicate_ids.append(case['id'])
            continue
        seen[key] = labels
        unique.append(case)
    (out / 'cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in unique))
    (out / 'knowledge.json').write_text(json.dumps(knowledge, ensure_ascii=False, indent=2)+'\n')
    write_json(out / 'manifest.json', {'scope': 'file-level authored and synthetic heldout fixtures',
        'original_cases': originals, 'generated_test_records': len(generated), 'unique_files': len(unique),
        'families': len({r['family'] for r in unique}),
        'benign_files': sum(not r['expected'] for r in unique), 'exact_duplicates_removed': duplicate_ids,
        'stage2_train_and_validation_family_overlap': 0,
        'source_sha256': digest(bundle / 'grpo_test.jsonl'), 'cases_sha256': digest(out / 'cases.jsonl'),
        'limitations': ['Synthetic labels need expert review', 'No independent benchmark certification',
                       'File-level input does not verify cross-file dataflow or runtime repair behavior']})
    print((out / 'manifest.json').read_text())


if __name__ == '__main__':
    main()
