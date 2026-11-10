"""Recompute paired financial retention from saved predictions and original labels.

CPU only. Never feeds evaluation content to a training dataset or changes model
selection. Confidence intervals describe saved question pairs, not an independent
confirmation of a candidate selected using the same validation set.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import digest, fingerprint, write_json
from fineval_pipeline.harmbench_comparison import paired_label_comparison
from scripts.eval_fineval import load_rows


def validate_predictions(directory, questions, data_sha):
    summary = json.loads((directory / 'summary.json').read_text())
    context = {'task': 'fineval-academic-val-option-logits-v2',
        'model': summary['model_identity'], 'data_sha256': data_sha, 'limit_per_subject': 0}
    expected = {(q['subject'], q['id']): q for q in questions}
    if len(expected) != len(questions):
        raise ValueError('Duplicate reference question identity')
    if (summary['task'] != context['task'] or summary['data_sha256'] != data_sha
            or summary['fingerprint'] != fingerprint(context)
            or summary['limit_per_subject'] != 0 or summary['examples'] != len(expected)
            or summary['model_identity']['precision'] != 'nf4'):
        raise ValueError('Wrong financial protocol, data, fingerprint, coverage or precision')
    if (directory / 'run.json').exists():
        saved = json.loads((directory / 'run.json').read_text())
        if saved != {'context': context, 'fingerprint': fingerprint(context)}:
            raise ValueError('Saved run context differs from summary and reference data')
    found = {}
    subjects = {}
    for line in (directory / 'predictions.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row['subject'], row['id']
        if key in found or key not in expected or row['fingerprint'] != summary['fingerprint']:
            raise ValueError('Duplicate, extra or wrong-model prediction')
        if row['prediction'] not in 'ABCD' or len(row['prediction']) != 1:
            raise ValueError('Invalid option prediction')
        correct = row['prediction'] == expected[key]['answer']
        if type(row['correct']) is not bool or row['correct'] != correct:
            raise ValueError('Saved correctness disagrees with original FinEval labels')
        found[key] = correct
        stat = subjects.setdefault(key[0], {'correct': 0, 'count': 0})
        stat['count'] += 1
        stat['correct'] += int(correct)
    if set(found) != set(expected):
        raise ValueError('Missing reference predictions')
    for stat in subjects.values():
        stat['accuracy'] = stat['correct'] / stat['count']
    if (subjects != summary['by_subject'] or summary['subjects'] != len(subjects)
            or abs(sum(found.values()) / len(found) - summary['micro_accuracy']) > 1e-12
            or abs(sum(s['accuracy'] for s in subjects.values()) / len(subjects)
                   - summary['macro_accuracy']) > 1e-12):
        raise ValueError('Saved aggregate scores disagree with original-label recomputation')
    return summary, found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    questions, data_sha = load_rows(args.data_dir, 0)
    if len(questions) != 1151:
        raise ValueError('Expected complete V2 Academic validation cohort of1151')
    before, left = validate_predictions(args.baseline, questions, data_sha)
    after, right = validate_predictions(args.candidate, questions, data_sha)
    if any(before['model_identity'][key] != after['model_identity'][key]
           for key in ['weights', 'precision', 'system_prompt']):
        raise ValueError('Financial comparison uses a different base, precision or system prompt')
    ids = [f'{q["subject"]}:{q["id"]}' for q in questions]
    left_labels = {f'{s}:{i}': [{'label': int(v)}] for (s, i), v in left.items()}
    right_labels = {f'{s}:{i}': [{'label': int(v)}] for (s, i), v in right.items()}
    pair = paired_label_comparison(left_labels, right_labels, ids)
    # The generic binary helper names successes ASR. Here successes are CORRECT
    # answers: explicitly translate all fields before publishing financial data.
    total = len(questions)
    result = {'state': 'completed', 'examples': total, 'subjects': len(before['by_subject']),
        'data_sha256': data_sha, 'baseline_identity': before['model_identity'],
        'candidate_identity': after['model_identity'], 'baseline_correct': sum(left.values()),
        'candidate_correct': sum(right.values()), 'baseline_accuracy': pair['baseline_asr'],
        'candidate_accuracy': pair['candidate_asr'], 'accuracy_change': pair['ASR_change'],
        'accuracy_change_percentage_points': 100 * pair['ASR_change'],
        'wrong_to_correct': pair['paired_regressed'], 'correct_to_wrong': pair['paired_improved'],
        'both_correct': pair['both_success'], 'both_wrong': pair['both_failure'],
        'paired_question_bootstrap95_accuracy_change': pair['paired_behavior_bootstrap95'],
        'exact_McNemar_two_sided_p': pair['exact_McNemar_two_sided_p'],
        'bootstrap_replicates': pair['bootstrap_replicates'], 'bootstrap_seed': pair['seed'],
        'retention_requirement': {'maximum_relative_score_drop': 0.05,
            'minimum_correct': math.ceil(0.95 * sum(left.values())),
            'passed': sum(right.values()) >= math.ceil(0.95 * sum(left.values()))},
        'by_subject': {s: {'baseline_correct': b['correct'],
            'candidate_correct': after['by_subject'][s]['correct'], 'count': b['count'],
            'change_correct': after['by_subject'][s]['correct'] - b['correct']}
            for s, b in before['by_subject'].items()},
        'evidence': {name: {filename: {'path': str(directory / filename),
            'sha256': digest(directory / filename)} for filename in ['summary.json', 'predictions.jsonl']}
            for name, directory in [('baseline', args.baseline), ('candidate', args.candidate)]},
        'scope': 'Full1151 GitHub V2 Academic val, custom option-logits protocol. '
            'Candidate was selected using this validation set; this is descriptive and not an independent '
            'financial improvement confirmation. Question bootstrap does not cover training randomness, '
            'subject clustering, or selection bias. No training, API calls, or GPU work.'}
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ['state', 'baseline_correct', 'candidate_correct',
        'wrong_to_correct', 'correct_to_wrong', 'accuracy_change_percentage_points',
        'paired_question_bootstrap95_accuracy_change', 'exact_McNemar_two_sided_p',
        'retention_requirement']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
