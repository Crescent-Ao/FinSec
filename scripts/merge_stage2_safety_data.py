"""Combine stage-2 safety sources while retaining their evaluation splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import write_json
from scripts.generate_stage2_deepseek import protected_index
from scripts.validate_pilot_data import normalized


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Use a new merged dataset version')
    source = args.generated.resolve()
    original = Path((ROOT / 'reports/latest_merged_full_run.txt').read_text().strip())
    legacy = original / 'supplemental_stages_v2'
    accepted, excluded = [], []
    source_counts = Counter()
    financial_tasks = {'finance_reasoning', 'finance_numeric'}
    for kind in ('sft_text', 'sft_tool', 'dpo', 'grpo'):
        for split in ('train', 'validation', 'test'):
            for row in rows(source / 'validated' / f'{kind}_{split}.jsonl'):
                if row['task_type'] in financial_tasks:
                    excluded.append({'id': row['id'], 'reason': 'standalone_finance_outside_stage2',
                                     'record_type': kind, 'split': split})
                    continue
                row['fixed_split'] = split
                row['merge_source'] = 'deepseek_stage2'
                accepted.append(row)
                source_counts['deepseek:' + kind + ':' + split] += 1
    guard = protected_index()
    for split in ('train', 'validation'):
        for kind, file in [('sft_text', legacy / 'safety' / f'sft_{split}.jsonl'),
                           ('dpo', legacy / 'safety' / f'dpo_{split}.jsonl'),
                           ('sft_tool', legacy / 'agent' / f'{split}.jsonl')]:
            for old in rows(file):
                if old.get('source') == 'replay_synth':
                    excluded.append({'id': old['id'], 'reason': 'financial_replay_outside_stage2',
                                     'record_type': kind, 'split': split})
                    continue
                text = ' '.join(m.get('content', '') for m in old['messages'])
                text += ' ' + old.get('rejected_response', '')
                matches = guard.search(normalized(text), protected=True)
                if matches:
                    excluded.append({'id': old['id'], 'reason': 'protected_reference_overlap',
                        'matches': [match['reference'] for match in matches[:3]], 'split': split})
                    continue
                origin = str(old.get('source', 'legacy_agent'))
                task = 'safety_boundary'
                if kind == 'sft_tool':
                    task = 'agent_tool_use'
                elif 'benign' in origin or 'anti_overrefusal' in origin:
                    task = 'benign_boundary'
                row = {'schema_version': 1, 'id': 'existing:' + old['id'], 'record_type': kind,
                    'family_id': 'existing:' + old['group_id'], 'parent_ids': [old['id']],
                    'task_type': task, 'language': 'zh', 'difficulty': 'medium', 'tags': ['existing_stage2'],
                    'fixed_split': split, 'merge_source': 'existing_stage2',
                    'provenance': old.get('provenance', {'source': origin}),
                    'review': {'status': 'accepted', 'checks': ['previous_preparation', 'protected_reference_guard'],
                               'issues': [], 'scope': 'teacher_semantics_require_sampling'}}
                if kind == 'dpo':
                    row.update(prompt=old['messages'][:-1], chosen=[old['messages'][-1]],
                        rejected=[{'role': 'assistant', 'content': old['rejected_response']}],
                        preference={'label': 'chosen', 'reason': 'Existing reviewed safety preference',
                                    'dimensions': ['safety', 'helpfulness'], 'error_tags': ['safety_boundary']})
                else:
                    row['messages'] = old['messages']
                    if old.get('tools'):
                        row['tools'] = old['tools']
                accepted.append(row)
                source_counts['existing:' + kind + ':' + split] += 1
    names = {'sft_text': 'sft_text', 'sft_tool': 'sft_tools', 'dpo': 'dpo_pairs', 'grpo': 'grpo_tasks'}
    for kind, name in names.items():
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
            for row in accepted if row['record_type'] == kind))
    (output / 'excluded.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in excluded))
    for name in ('artifacts', 'execution_logs', 'verifiers'):
        shutil.copytree(source / name, output / name)
    manifest = {'status': 'merged_pending_native_validation', 'training_scope': 'stage2_safety_only',
        'first_stage_financial_training_included': False,
        'sources': {'generated': str(source), 'existing_stage2': str(legacy)},
        'source_counts': dict(source_counts), 'counts': dict(Counter(row['record_type'] for row in accepted)),
        'excluded': dict(Counter(row['reason'] for row in excluded)),
        'split_policy': 'preserve fixed heldouts; merge near families with test then validation priority',
        'files': {str(path.relative_to(output)): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'bytes': path.stat().st_size} for path in output.rglob('*') if path.is_file()}}
    write_json(output / 'manifest.json', manifest)
    (output / 'README.md').write_text('# 第二阶段安全增强数据\n\n'
        '合并安全边界、抗误拒、隐私行为与密码学审计训练来源。金融能力使用 FinEval 监测。\n\n'
        '各原始集合的评测身份保留，统一近重复家族验证后形成最终训练文件。\n')
    print(json.dumps({'counts': manifest['counts'], 'excluded': manifest['excluded'],
                      'source_counts': manifest['source_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
