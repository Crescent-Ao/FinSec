from .core import append, bind, digest, resolve, rows, write_json
from scripts.eval_fineval import load_rows
import json


def verify_reference(cfg):
    directory=resolve(cfg['fineval']['data_dir'])
    manifest=json.loads(resolve(cfg['fineval'].get('source_manifest','reports/fineval_guard_manifest.json')).read_text())
    expected={r['file'].split('/')[-1]:r for r in manifest['csv_files'] if r['file'].startswith('val/')}
    actual={p.name:p for p in directory.glob('*.csv')}
    if set(expected)!=set(actual):
        raise ValueError('FinEval subject coverage differs from the pinned official reference')
    for name,path in actual.items():
        if digest(path)!=expected[name]['sha256']:
            raise ValueError('FinEval reference CSV changed: '+name)
    return manifest


def evaluate(cfg, name, backend, identity, directory):
    settings = cfg['fineval']
    if settings['split'] != 'val':
        raise ValueError('Local Academic test has no answer labels; cannot fabricate a test score')
    verify_reference(cfg)
    questions, data_hash = load_rows(resolve(settings['data_dir']), settings['limit_per_subject'])
    context = {'task': 'fineval-academic-val-option-logits-v2', 'model': identity, 'data_sha256': data_hash,
               'limit_per_subject': settings['limit_per_subject']}
    run_id = bind(directory, context)
    path = directory/'predictions.jsonl'
    saved = rows(path) if path.exists() else []
    done = {(r['subject'], r['id']): r for r in saved}
    if len(done) != len(saved) or any(r['fingerprint']!=run_id for r in saved):
        raise ValueError('Invalid FinEval resume records')
    for q in questions:
        key = (q['subject'], q['id'])
        if key in done:
            continue
        prompt = '请回答以下单项选择题，仅输出正确选项字母A、B、C或D。\n'+q['question']+'\n'
        prompt += '\n'.join(f"{k}. {q['options'][k]}" for k in 'ABCD')
        prediction, scores = backend.options(prompt)
        result = {'subject': q['subject'], 'id': q['id'], 'prediction': prediction,
                  'correct': prediction==q['answer'], 'scores': scores, 'fingerprint': run_id}
        append(path, result)
        done[key] = result
    subjects = {}
    for q in questions:
        s = subjects.setdefault(q['subject'], {'correct': 0, 'count': 0})
        s['correct'] += int(done[(q['subject'], q['id'])]['correct'])
        s['count'] += 1
    for stat in subjects.values():
        stat['accuracy'] = stat['correct']/stat['count']
    summary = {'task': context['task'], 'fingerprint': run_id, 'model_identity': identity,
               'data_sha256': data_hash, 'examples': len(questions), 'subjects': len(subjects),
               'micro_accuracy': sum(s['correct'] for s in subjects.values())/len(questions),
               'macro_accuracy': sum(s['accuracy'] for s in subjects.values())/len(subjects), 'by_subject': subjects,
               'limit_per_subject': settings['limit_per_subject'],
               'scope': 'smoke_subset' if settings['limit_per_subject'] else 'academic_val_custom_protocol_not_official_test'}
    write_json(directory/'summary.json', summary)
    return summary
