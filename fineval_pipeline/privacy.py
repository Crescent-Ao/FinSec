import json
import math
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .core import ROOT, append, bind, digest, fingerprint, resolve, rows, write_json


def training_input_path(value):
    """Locate an unchanged training snapshot after moving the project to A100."""
    actual=resolve(value)
    if not actual.exists() and '/outputs/' in str(actual):
        actual=ROOT/'outputs'/str(actual).split('/outputs/',1)[1]
    return actual


def attack_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('MIA metrics require finite scores and both classes')
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(fpr <= 0.01 + 1e-12)
    best = candidates[np.argmax(tpr[candidates])]
    return {'roc_auc': float(roc_auc_score(labels, scores)), 'tpr_at_1pct_fpr': float(tpr[best]),
            'achieved_fpr': float(fpr[best]),
            'threshold': float(thresholds[best]) if np.isfinite(thresholds[best]) else None,
            'score_direction': 'higher means more likely member; no post-hoc sign flip'}


def bootstrap_ci(records, key, repeats, seed):
    rng = np.random.default_rng(seed)
    strata = []
    for label in (0, 1):
        groups = {}
        for row in records:
            if row['member'] == label:
                groups.setdefault(row['group_id'], []).append(row)
        strata.append(list(groups.values()))
    values = []
    for _ in range(repeats):
        sample = []
        for groups in strata:
            for index in rng.integers(len(groups), size=len(groups)):
                sample.extend(groups[index])
        values.append(attack_metrics([r['member'] for r in sample], [r[key] for r in sample]))
    if not values:
        return None
    return {metric: [float(x) for x in np.percentile([v[metric] for v in values], [2.5, 97.5])]
            for metric in ('roc_auc', 'tpr_at_1pct_fpr')}


def membership_data(cfg):
    settings = cfg['privacy']
    if settings.get('selection'):
        return selected_membership_data(cfg)
    members, nonmembers = rows(resolve(settings['members'])), rows(resolve(settings['nonmembers']))
    manifest = json.loads(resolve(cfg['models']['M1']['training_manifest']).read_text())
    state = json.loads(resolve(cfg['models']['M1']['training_state']).read_text())
    if float(state['epoch']) < 1:
        raise ValueError('Partial-epoch run needs an actual exposed-ID ledger; planned train IDs are insufficient')
    args_path = resolve(cfg['models']['M1']['training_state']).parent/'args.json'
    training_args = json.loads(args_path.read_text())
    datasets = training_args['dataset']
    if len(datasets)!=1:
        raise ValueError('Multiple training inputs require a combined actual-exposure manifest')
    from pathlib import Path
    actual = Path(datasets[0])
    if not actual.exists() and '/outputs/' in str(actual):
        actual = ROOT/'outputs'/str(actual).split('/outputs/',1)[1]
    if digest(actual) != manifest['splits']['train']['training_file_sha256']:
        raise ValueError('The actually consumed training file differs from the membership ledger')
    if [fingerprint(r['messages']) for r in rows(actual)] != [fingerprint(r['messages']) for r in members]:
        raise ValueError('Training messages differ from the claimed member contents/order')
    for split, records, file in [('train', members, settings['members']),
                                  ('privacy_nonmembers', nonmembers, settings['nonmembers'])]:
        expected = manifest['splits'][split]
        if set(expected['ids']) != {r['id'] for r in records} or len(expected['ids']) != len(records):
            raise ValueError(f'Membership ID mismatch: {split}')
        if digest(resolve(file)) != expected['source_sha256']:
            raise ValueError(f'Membership content mismatch: {split}')
    if ({r['id'] for r in members} & {r['id'] for r in nonmembers} or
        {r['group_id'] for r in members} & {r['group_id'] for r in nonmembers} or
        {fingerprint(r['messages']) for r in members} & {fingerprint(r['messages']) for r in nonmembers}):
        raise ValueError('Member/nonmember leakage detected')
    limit = settings.get('limit_per_class', 0)
    selected = [(r, 1) for r in (members[:limit] if limit else members)]
    selected += [(r, 0) for r in (nonmembers[:limit] if limit else nonmembers)]
    return selected, manifest


def selected_membership_data(cfg):
    """Verify the entire exposure ledger while retaining only fixed evaluation IDs."""
    from itertools import zip_longest
    settings = cfg['privacy']; registration = cfg['models']['M1']
    manifest = json.loads(resolve(registration['training_manifest']).read_text())
    selection_path = resolve(settings['selection'])
    if digest(selection_path) != manifest['privacy_selection_sha256']:
        raise ValueError('Privacy sample selection changed')
    selection = json.loads(selection_path.read_text())
    member_ids, nonmember_ids = selection['member_ids'], selection['nonmember_ids']
    if (not member_ids or not nonmember_ids or len(member_ids) != len(set(member_ids)) or
        len(nonmember_ids) != len(set(nonmember_ids)) or set(member_ids) & set(nonmember_ids)):
        raise ValueError('Invalid fixed privacy selection')
    state_path = resolve(registration['training_state'])
    state = json.loads(state_path.read_text())
    if float(state['epoch']) < 1:
        raise ValueError('Full membership ledger requires one complete training epoch')
    training_args = json.loads((state_path.parent/'args.json').read_text())
    if len(training_args['dataset']) != 1:
        raise ValueError('Expected a single frozen training file')
    actual = training_input_path(training_args['dataset'][0])
    if digest(actual) != manifest['splits']['train']['training_file_sha256']:
        raise ValueError('Actual training input differs from the full exposure ledger')
    def stream(path):
        with resolve(path).open() as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    retained = {}; private_messages = set(); private_groups = set(); private_ids = set()
    for split, file, chosen in [('privacy_nonmembers', settings['nonmembers'], set(nonmember_ids)),
                                 ('train', settings['members'], set(member_ids))]:
        expected = manifest['splits'][split]
        if digest(resolve(file)) != expected['source_sha256']:
            raise ValueError('Membership file hash mismatch: '+split)
        remaining = set(expected['ids']); allowed_groups = set(expected['groups'])
        iterator = stream(file)
        paired = zip_longest(iterator, stream(actual)) if split == 'train' else ((r, None) for r in iterator)
        for row, consumed in paired:
            if row is None or row['id'] not in remaining:
                raise ValueError('Missing or duplicate membership ID: '+split)
            remaining.remove(row['id'])
            if row['group_id'] not in allowed_groups:
                raise ValueError('Unexpected membership family')
            content = fingerprint(row['messages'])
            if split == 'train':
                if consumed is None or row['messages'] != consumed['messages']:
                    raise ValueError('Training file contents/order differ from membership ledger')
                if row['id'] in private_ids or row['group_id'] in private_groups or content in private_messages:
                    raise ValueError('Member/nonmember leakage')
            else:
                private_ids.add(row['id']); private_groups.add(row['group_id']); private_messages.add(content)
            if row['id'] in chosen: retained[row['id']] = (row, int(split == 'train'))
        if remaining: raise ValueError('Incomplete membership ledger: '+split)
    ordered = member_ids+nonmember_ids
    if any(identity not in retained for identity in ordered):
        raise ValueError('A selected privacy sample is absent from its frozen split')
    return [retained[identity] for identity in ordered], manifest


def evaluate(cfg, name, backend, identity, directory, root):
    selected, manifest = membership_data(cfg)
    settings = cfg['privacy']
    k = settings['min_k_fraction']
    if not 0 < k <= 1:
        raise ValueError('min_k_fraction must be in (0,1]')
    context = {'task': 'membership-inference-answer-logprob-v1', 'model': identity,
               'members': digest(resolve(settings['members'])),
               'nonmembers': digest(resolve(settings['nonmembers'])), 'k': k,
               'selected_ids': [r['id'] for r, _ in selected]}
    run_id = bind(directory, context)
    path = directory/'scores.jsonl'
    done = {r['id']: r for r in rows(path)} if path.exists() else {}
    if len(done) != (len(rows(path)) if path.exists() else 0):
        raise ValueError('Duplicate privacy checkpoint records')
    for row, member in selected:
        if row['id'] in done:
            if done[row['id']]['fingerprint'] != run_id:
                raise ValueError('MIA checkpoint fingerprint mismatch')
            continue
        lp = np.asarray(backend.answer_logprobs(row['messages']))
        result = {'id': row['id'], 'group_id': row['group_id'], 'domain': row['domain'],
                  'member': member, 'answer_tokens': len(lp), 'loss_score': float(lp.mean()),
                  'min_k_score': float(np.sort(lp)[:max(1, math.ceil(k*len(lp)))].mean()),
                  'fingerprint': run_id}
        append(path, result)
        done[row['id']] = result
        print(f'privacy {name}: {len(done)}/{len(selected)}', flush=True)
    records = [done[r['id']] for r, _ in selected]
    reference = root/'M0/privacy/scores.jsonl'
    if name != 'M0' and reference.exists():
        from .core import model_identity, model_spec
        reference_context = json.loads((reference.parent/'run.json').read_text())['context']
        for key in ('task','members','nonmembers','selected_ids','k'):
            if reference_context[key] != context[key]:
                raise ValueError('Reference M0 privacy protocol/data mismatch')
        if reference_context['model'] != model_identity(model_spec(cfg,'M0')):
            raise ValueError('Reference M0 scores belong to an old base model')
        base = {r['id']: r for r in rows(reference)}
        for r in records:
            if r['id'] not in base:
                raise ValueError('M0 reference sample missing')
            # loss(M0) - loss(target) == logprob(target) - logprob(M0)
            r['reference_score'] = r['loss_score']-base[r['id']]['loss_score']
    attacks = {}
    keys = ['loss_score', 'min_k_score'] + (['reference_score'] if 'reference_score' in records[0] else [])
    for key in keys:
        attacks[key] = attack_metrics([r['member'] for r in records], [r[key] for r in records])
        attacks[key]['family_bootstrap_95pct_ci'] = bootstrap_ci(records, key, settings['bootstrap_replicates'], cfg['seed'])
    nonmember_count = sum(r['member']==0 for r in records)
    summary = {'task': context['task'], 'fingerprint': run_id, 'model_identity': identity,
               'attacks': attacks, 'members': len(records)-nonmember_count, 'nonmembers': nonmember_count,
               'fpr_resolution': 1/nonmember_count, 'low_fpr_reliability': 'pilot_only' if nonmember_count < 1000 else 'report_CI',
               'scope': 'fixed_stratified_actual_training_sample' if settings.get('selection') else 'smoke_subset' if settings.get('limit_per_class') else 'actual_stage1_training_members',
               'selection_manifest_sha256': digest(resolve(settings['selection'])) if settings.get('selection') else None,
               'domain_counts': dict(Counter((str(r['member'])+':'+r['domain']) for r in records)),
               'selected_source_counts': dict(Counter(str(label)+':'+row.get('source_name','legacy') for row,label in selected)),
               'answer_tokens_mean': {str(y): float(np.mean([r['answer_tokens'] for r in records if r['member']==y])) for y in (0, 1)},
               'serialization': 'native nonthinking chat prefix + original assistant answer, no EOS; answer tokens only',
               'reference_attack_available': 'reference_score' in records[0]}
    summary['bootstrap_replicates']=settings['bootstrap_replicates']
    summary['bootstrap_seed']=cfg['seed']
    write_json(directory/'summary.json', summary)
    return summary
