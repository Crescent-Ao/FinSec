"""CPU-only verification of HarmBench's official classifier-tokenizer clipping.

The upstream evaluate_completions.py re-tokenizes and right-clips each native
answer to512 classifier tokens. Its saved generation is therefore not always
byte-identical to the target model's512-token answer. Reproduce the exact
upstream transformation instead of accepting an arbitrary text difference.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import digest, fingerprint, write_json


def verify_transformation(completions, labels, cases, tokenizer, *, token_limit=512):
    if set(completions) != set(labels) or set(completions) != set(cases):
        raise ValueError('Scoring source and official result behavior IDs differ')
    changed = 0
    for behavior, rows in completions.items():
        if len(rows) != 1 or len(labels[behavior]) != 1 or len(cases[behavior]) != 1:
            raise ValueError('Expected exactly one native case, answer and label per behavior')
        original, official = rows[0], labels[behavior][0]
        if original['test_case'] != official['test_case'] or original['test_case'] != cases[behavior][0]:
            raise ValueError('Scoring result has a different original attack case')
        ids = tokenizer.encode(original['generation'], max_length=token_limit, truncation=True)
        expected = tokenizer.decode(ids, skip_special_tokens=True)
        if official['generation'] != expected:
            raise ValueError('Saved classifier input differs from exact upstream tokenizer clipping')
        changed += official['generation'] != original['generation']
    return {'behaviors_verified': len(completions), 'generation_text_changed_by_official_tokenization': changed,
        'generation_text_unchanged': len(completions) - changed,
        'all_case_links_and_official_clipped_generations_match': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--classifier-tokenizer', type=Path, required=True)
    parser.add_argument('--classifier-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.directory
    summary = json.loads((folder / 'summary.json').read_text())
    if summary['classifier_identity']['manifest_sha256'] != digest(args.classifier_manifest):
        raise ValueError('Official classifier manifest differs from saved score')
    manifest = json.loads(args.classifier_manifest.read_text())
    if (manifest['repo'] != 'cais/HarmBench-Llama-2-13b-cls'
            or manifest['revision'] != 'bda705349d1144fa618770bea64d99ce54e3835b'):
        raise ValueError('Requires the pinned official test classifier')
    tokenizer_files = []
    for row in manifest['files']:
        filename = row['file']
        if filename.endswith(('.json', '.model', '.txt', '.jinja')):
            if digest(args.classifier_tokenizer / filename) != row['sha256']:
                raise ValueError('Classifier tokenizer/config bytes changed')
            tokenizer_files.append(row)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_tokenizer, local_files_only=True)
    tokenizer.truncation_side = 'right'
    completions = json.loads((folder / 'completions.json').read_text())
    labels = json.loads((folder / 'official_results.json').read_text())
    cases = json.loads((folder / 'test_cases/test_cases.json').read_text())
    if summary['behaviors'] != 320 or len(completions) != 320:
        raise ValueError('Formal evidence requires all320 behaviors')
    if digest(folder / 'official_results.json') != summary['official_results_sha256']:
        raise ValueError('Official label bytes changed')
    for filename, key in [('completions.json', 'completions_sha256'), ('behaviors.csv', 'behaviors_sha256')]:
        if digest(folder / filename) != summary['judge_protocol'][key]:
            raise ValueError('Official classifier source bytes changed')
    context = json.loads((folder / 'judge_run/run.json').read_text())
    if context['fingerprint'] != fingerprint(context['context']) or context['context'] != summary['judge_protocol']:
        raise ValueError('Saved official judge context differs')
    behavior_ids = [r['BehaviorID'] for r in csv.DictReader((folder / 'behaviors.csv').open())]
    if len(behavior_ids) != 320 or len(set(behavior_ids)) != 320 or set(behavior_ids) != set(completions):
        raise ValueError('Classifier behavior reference differs')
    result = verify_transformation(completions, labels, cases, tokenizer)
    result.update(state='completed', classifier_repo=manifest['repo'], classifier_revision=manifest['revision'],
        target_max_new_tokens=summary['attack']['target_max_new_tokens'], classifier_input_token_limit=512,
        classifier_tokenizer_files=tokenizer_files, official_results_sha256=digest(folder / 'official_results.json'),
        completions_sha256=digest(folder / 'completions.json'), audit_source_sha256=digest(__file__),
        score_recomputed_or_changed=False, GPU_or_API_work=False,
        scope='Exact native case→answer→official512-classifier-token input transformation, no label/model modification')
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ['state', 'behaviors_verified',
        'generation_text_changed_by_official_tokenization', 'generation_text_unchanged',
        'all_case_links_and_official_clipped_generations_match']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
