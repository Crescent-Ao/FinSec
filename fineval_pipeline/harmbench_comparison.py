"""Paired, coverage-strict comparison of single-case official HarmBench labels."""
import math

import numpy as np


def wilson_interval(successes, total):
    if not 0 <= successes <= total or total < 1:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., midpoint - half_width), min(1., midpoint + half_width)]


def paired_label_comparison(baseline, candidate, expected_ids, *, bootstrap_replicates=2000, seed=20261109):
    ids = list(expected_ids)
    if not ids or len(ids) != len(set(ids)) or set(baseline) != set(ids) or set(candidate) != set(ids):
        raise ValueError('Paired ASR comparison requires complete, identical behavior coverage')
    if bootstrap_replicates < 100:
        raise ValueError('Too few replicates for the declared paired bootstrap interval')
    before, after = [], []
    for key in ids:
        for records, destination in [(baseline[key], before), (candidate[key], after)]:
            if len(records) != 1 or type(records[0].get('label')) not in (int, bool) or records[0]['label'] not in (0, 1):
                raise ValueError('Paired binary ASR expects exactly one valid official label per behavior')
            destination.append(int(records[0]['label']))
    before, after = np.asarray(before), np.asarray(after)
    improved = int(((before == 1) & (after == 0)).sum())
    regressed = int(((before == 0) & (after == 1)).sum())
    discordant = improved + regressed
    # Exact two-sided McNemar is a binomial sign test over discordant pairs.
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2 ** discordant) if discordant else 1.
    rng = np.random.default_rng(seed)
    differences = (after - before).astype(float)
    sampled = rng.integers(0, len(ids), size=(bootstrap_replicates, len(ids)))
    changes = differences[sampled].mean(axis=1)
    n = len(ids)
    return {'behaviors': n, 'baseline_successes': int(before.sum()), 'candidate_successes': int(after.sum()),
        'baseline_asr': float(before.mean()), 'candidate_asr': float(after.mean()),
        'baseline_wilson95': wilson_interval(int(before.sum()), n),
        'candidate_wilson95': wilson_interval(int(after.sum()), n),
        'ASR_change': float(differences.mean()),
        'paired_behavior_bootstrap95': [float(x) for x in np.quantile(changes, [.025, .975])],
        'paired_improved': improved, 'paired_regressed': regressed,
        'both_success': int(((before == 1) & (after == 1)).sum()),
        'both_failure': int(((before == 0) & (after == 0)).sum()),
        'exact_McNemar_two_sided_p': pvalue,
        'bootstrap_replicates': bootstrap_replicates, 'seed': seed,
        'scope': 'uncertainty across the supplied behavior sample conditional on one saved search/seed; does not cover attack randomness, classifier error or development selection bias'}
