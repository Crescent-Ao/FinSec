"""Paired, family-resampled comparisons for the frozen stage-one MIA cohort."""
import numpy as np

from .privacy import attack_metrics


def align_scores(models):
    aligned = {}
    expected = None
    for name, records in models.items():
        index = {row['id']: row for row in records}
        if len(index) != len(records):
            raise ValueError('Duplicate MIA score IDs: ' + name)
        if expected is None:
            expected = list(index)
        if set(index) != set(expected):
            raise ValueError('MIA comparison samples differ: ' + name)
        aligned[name] = [index[key] for key in expected]
    reference = next(iter(aligned.values()))
    for name, records in aligned.items():
        for a, b in zip(reference, records):
            if any(a[key] != b[key] for key in ('member', 'group_id', 'answer_tokens')):
                raise ValueError('MIA sample labels/families/serialization differ: ' + name)
    return aligned


def paired_changes(models, comparisons, repeats=500, seed=20261109):
    aligned = align_scores(models)
    reference = next(iter(aligned.values()))
    labels = np.array([row['member'] for row in reference])
    fields = ['loss_score', 'min_k_score']
    scores = {name: {field: np.array([row[field] for row in records]) for field in fields}
              for name, records in aligned.items()}
    base = scores['M0']['loss_score']
    for name in scores:
        if name != 'M0':
            scores[name]['reference_score'] = scores[name]['loss_score'] - base
    groups = []
    for label in (0, 1):
        families = {}
        for index, row in enumerate(reference):
            if row['member'] == label:
                families.setdefault(row['group_id'], []).append(index)
        groups.append([np.array(indices) for indices in families.values()])
    point = {name: {field: attack_metrics(labels, values) for field, values in attacks.items()}
             for name, attacks in scores.items()}
    metrics = ['roc_auc', 'tpr_at_1pct_fpr']
    distributions = {}
    output = {}
    for title, before, after in comparisons:
        shared = set(scores[before]) & set(scores[after])
        output[title] = {'before': before, 'after': after, 'attacks': {}}
        distributions[title] = {field: {metric: [] for metric in metrics} for field in shared}
        for field in sorted(shared):
            output[title]['attacks'][field] = {
                metric: {'delta_after_minus_before': point[after][field][metric] - point[before][field][metric]}
                for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        indices = np.concatenate([family_groups[i] for family_groups in groups
                                  for i in rng.integers(len(family_groups), size=len(family_groups))])
        values = {name: {field: attack_metrics(labels[indices], values[indices])
                        for field, values in attacks.items()} for name, attacks in scores.items()}
        for title, before, after in comparisons:
            for field, measures in distributions[title].items():
                for metric, samples in measures.items():
                    samples.append(values[after][field][metric] - values[before][field][metric])
    for title, fields in distributions.items():
        for field, measures in fields.items():
            for metric, samples in measures.items():
                interval = np.percentile(samples, [2.5, 97.5]).tolist()
                output[title]['attacks'][field][metric].update(
                    paired_family_bootstrap_95pct_ci=interval,
                    direction='lower_after' if interval[1] < 0 else 'higher_after' if interval[0] > 0 else 'inconclusive')
    return {'comparisons': output, 'bootstrap_replicates': repeats, 'bootstrap_seed': seed,
            'method': 'same resampled families for all models; separate member/nonmember strata',
            'scope': 'fixed fine-tuning cohort; no claim about pretraining membership or a privacy guarantee'}
