"""Adopt the independently verified native batch8 NF4 replica implementation."""
import json
from pathlib import Path
import shutil

from .core import ROOT, digest, write_json


def verify_adoption(root, identity, runtime):
    hard = root / 'outputs/security_hardening_20261109_v1'
    policy_path = hard / 'native_pool_adoption.json'
    policy = json.loads(policy_path.read_text())
    if policy.get('enabled') is not True or identity not in policy['allowed_model_identities']:
        raise ValueError('Native pool adoption does not cover this checkpoint')
    proof = hard / 'GCG_completion_replication_full320'
    result = json.loads((proof / 'result.json').read_text())
    plan = json.loads((proof / 'plan.json').read_text())
    if (digest(proof / 'result.json') != policy['proof_result_sha256']
            or digest(proof / 'plan.json') != policy['proof_plan_sha256']
            or result['source_plan_sha256'] != digest(proof / 'plan.json')):
        raise ValueError('Verified native replication proof bytes changed')
    if (result['behaviors'] != 320 or result['records_identical_to_reference'] != 320
            or result['all320_case_and_answer_records_identical_to_reference'] is not True
            or result['whole_completion_file_byte_identical_to_reference'] is not True
            or result['model_identity']['weights'] != identity['weights']
            or result['model_identity']['precision'] != identity['precision']
            or identity['precision'] != 'nf4'):
        raise ValueError('Pool adoption needs complete matching NF4 baseline evidence')
    reference = hard / 'formal_finance_scoring/evaluation/M1/harmbench/GCG/completions.json'
    if (digest(proof / 'completions.json') != digest(reference)
            or digest(reference) != result['completions_sha256']):
        raise ValueError('Full320 native answers no longer equal the scored baseline')
    records = json.loads(reference.read_text())
    if len(records) != 320 or any(len(rows) != 1 for rows in records.values()):
        raise ValueError('Recorded equivalence must cover320 actual baseline answers')
    for file, key in [(root / 'scripts/verify_gcg_completion_replication.py', 'pool_source_sha256'),
                      (root / 'scripts/probe_gcg_completion_replicas.py', 'native_worker_sha256'),
                      (runtime / 'generate_completions.py', 'native_source_sha256')]:
        if digest(file) != plan[key] or digest(file) != policy[key]:
            raise ValueError('Inference code differs from the full320 verified implementation')
    return {'policy_sha256': digest(policy_path), 'full320_result_sha256': digest(proof / 'result.json'),
        'full320_plan_sha256': digest(proof / 'plan.json'), 'reference_completions_sha256': digest(reference)}


def generate_target_answers(cfg, runtime, models, behaviors, directory, identity, *, allow_pool=True):
    """Caller already holds the A100 lease for inference through official scoring."""
    from .harmbench_upstream import run_command
    policy_path = ROOT / 'outputs/security_hardening_20261109_v1/native_pool_adoption.json'
    cases = directory / 'test_cases/test_cases.json'
    target = directory / 'completions.json'
    if not allow_pool or not policy_path.exists():
        run_command(cfg, runtime, ['generate_completions.py', '--model_name', 'target',
            '--models_config_file', str(models), '--behaviors_path', str(behaviors),
            '--test_cases_path', str(cases), '--save_path', str(target),
            '--incremental_update', '--max_new_tokens', '512'], directory / 'step2_completions.log')
        return {'mode': 'native_HF_serial', 'logical_batch_cap': 8, 'max_new_tokens': 512}
    evidence = verify_adoption(ROOT, identity, runtime)
    from scripts.verify_gcg_completion_replication import run_native_pool
    pool = directory / 'native_pool'
    saved = pool / 'result.json'
    if saved.exists():
        result = json.loads(saved.read_text())
        plan = json.loads((pool / 'plan.json').read_text())
        if (result['state'] != 'completed' or result['model_identity'] != identity
                or digest(cases) != plan['cases_file_sha256'] or digest(models) != plan['models_file_sha256']
                or digest(pool / 'completions.json') != result['completions_sha256']
                or digest(pool / 'plan.json') != result['source_plan_sha256']):
            raise ValueError('Saved native pool generation does not match current inputs')
    else:
        result = run_native_pool(pool, runtime=runtime, models_file=models, cases_file=cases, identity=identity)
    if result['behaviors'] != 320 or result['replicas'] != 4 or result['batch_each'] != 8 or result['max_new_tokens'] != 512:
        raise ValueError('Native pool output coverage or generation parameters changed')
    if target.exists() and digest(target) != result['completions_sha256']:
        raise ValueError('Preserve existing different native answers; do not overwrite')
    if not target.exists():
        shutil.copyfile(pool / 'completions.json', target)
    execution = {'mode': 'native_HF_four_replicas', 'starting_batch_cap': 8, 'replicas': 4,
        'max_new_tokens': 512, 'do_sample': False, 'model_identity': identity,
        'upstream_script_sha256': digest(runtime / 'generate_completions.py'),
        'pool_result_sha256': digest(saved), 'pool_plan_sha256': digest(pool / 'plan.json'),
        'completions_sha256': digest(target), 'adoption_evidence': evidence,
        'adopted_for_official_scoring': True,
        'change': 'Schedule original batch8 cohorts across four unmerged NF4 replicas; reuse native HF generation',
        'limitation': 'Full320 equivalence was verified on the financial baseline; not a proof of all other checkpoints'}
    write_json(directory / 'completions_execution.json', execution)
    return execution
