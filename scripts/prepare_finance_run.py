"""Freeze a small, explicitly synthetic pilot SFT experiment after validation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    quality = json.loads((ROOT / 'reports/pilot_luna_v1_quality.json').read_text())
    assert quality['status'] == 'candidate_checks_passed'
    assert quality['fineval_questions_checked'] == 4661
    data = ROOT / 'data/pilot_luna_v1'
    reviews = {}
    for owner, name in [('finance', 'finance_final_review.json'),
                        ('accounting', 'accounting_final_review.json'),
                        ('economics_professional', 'economics_professional_resolutions.json')]:
        path = data / 'parts' / name
        review = json.loads(path.read_text())
        records = [json.loads(x) for x in (data / 'parts' / f'{owner}.jsonl').read_text().splitlines()]
        assert set(review['reviewed_ids']) == {r['id'] for r in records}, name
        reviewed_hash = review.get('reviewed_parts_sha256', review.get('reviewed_snapshot_sha256'))
        assert reviewed_hash == sha(data / 'parts' / f'{owner}.jsonl'), 'Review snapshot mismatch'
        reviews[name] = sha(path)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen3.5-9B', local_files_only=True)
    run = ROOT / 'outputs' / ('finance_pilot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(parents=True)
    manifest = {'scope': 'synthetic pilot; LLM reviewed, not expert reviewed',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'reviews': reviews,
                'base_model_manifest_sha256': sha(ROOT / 'reports/model_manifest.json'),
                'quality_report_sha256': sha(ROOT / 'reports/pilot_luna_v1_quality.json'),
                'splits': {}, 'seed': 42, 'max_length': 512,
                'privacy_policy': 'Only train IDs are members. Nonmembers excluded from all training and retrieval.'}
    groups = set()
    for split, expected in [('train', 300), ('validation', 100), ('privacy_nonmembers', 100)]:
        source = data / f'{split}.candidates.jsonl'
        rows = [json.loads(x) for x in source.read_text().splitlines()]
        assert len(rows) == expected
        split_groups = {r['group_id'] for r in rows}
        assert not groups & split_groups
        groups.update(split_groups)
        lengths = [len(tokenizer.apply_chat_template(r['messages'], enable_thinking=False, return_dict=False)) for r in rows]
        assert max(lengths) <= 512, 'Increase reviewed max_length instead of silently truncating.'
        manifest['splits'][split] = {'rows': len(rows), 'ids': [r['id'] for r in rows],
                                    'source_sha256': sha(source), 'max_tokens': max(lengths)}
        if split != 'privacy_nonmembers':
            target = run / f'{split}.jsonl'
            target.write_text(''.join(json.dumps({'messages': r['messages']}, ensure_ascii=False) + '\n' for r in rows))
            manifest['splits'][split]['training_file_sha256'] = sha(target)
    (run / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(run)


if __name__ == '__main__':
    main()
