"""Validate a full PAIR search handoff before native target generation and judging."""
import copy


def portable_parameters(parameters, targets_sha256):
    params = copy.deepcopy(parameters)
    params.pop('target_model', None)
    params.pop('dependencies', None)
    params['targets_path'] = 'sha256:' + targets_sha256
    for name in ('attack_model', 'judge_model'):
        params[name].pop('api_variant', None)
    return params


def validate_import(receipt, cases, *, identity, behavior_ids, behaviors_sha256,
                    targets_sha256, parameters, core_sha256, revision, seed,
                    cases_sha256, allow_common_subset=False):
    """Reject incomplete, mismatched or empty search outputs, never infer zero ASR."""
    if receipt.get('schema') != 1 or receipt.get('method') != 'PAIR':
        raise ValueError('Unknown PAIR handoff schema or method')
    expected = list(behavior_ids)
    selected = receipt.get('selected_ids', [])
    if len(expected) != 320 or len(set(expected)) != 320:
        raise ValueError('Formal PAIR scoring requires the fixed 320 test behaviors')
    if allow_common_subset:
        if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(expected):
            raise ValueError('PAIR common subset must be a nonempty unique subset of the fixed cohort')
    elif len(selected) != 320 or len(set(selected)) != 320 or set(selected) != set(expected):
        raise ValueError('PAIR search handoff coverage differs from all 320 test behaviors')
    if not isinstance(cases, dict) or set(cases) != set(selected):
        raise ValueError('PAIR test cases do not cover the full fixed cohort')
    for rows in cases.values():
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], str) or not rows[0].strip():
            raise ValueError('Each behavior requires exactly one nonempty PAIR search output')
    fields = {
        'model_identity': identity, 'behaviors_sha256': behaviors_sha256,
        'targets_sha256': targets_sha256,
        'parameters': portable_parameters(parameters, targets_sha256),
        'official_PAIR_core_sha256': core_sha256, 'upstream_revision': revision,
        'seed': seed, 'test_cases_sha256': cases_sha256,
        'api_model': 'deepseek-flash', 'thinking': False,
    }
    for key, expected_value in fields.items():
        if receipt.get(key) != expected_value:
            raise ValueError('PAIR source identity or protocol differs: ' + key)
    if identity.get('precision') != 'nf4':
        raise ValueError('PAIR target scoring must keep the NF4 baseline')
    return {'verified_behaviors': len(selected), 'requested_behaviors':320, 'full_coverage':len(selected)==320, 'target_identity': identity,
            'scope': 'validated search handoff only; no ASR until native generation and official scoring'}
