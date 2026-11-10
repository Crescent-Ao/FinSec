"""Native NF4 batch8 replication; default is a full320 baseline equivalence trial.

Only inference is parallelized. Original attack cases and scores are immutable.
The public generation function assumes its caller already holds the A100 lease.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import bind, digest, fingerprint, write_json
from scripts import probe_gcg_completion_replicas as native_probe


def assignments_for_cases(cases):
    records = []
    for behavior, values in cases.items():
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError('Requires one text attack per behavior')
        records.append({'behavior_id': behavior, 'test_case': values[0]})
    if len(records) != 320:
        raise ValueError('Full320 behavior coverage is required')
    groups = [[] for _ in range(4)]
    for i in range(0, len(records), 8):
        groups[(i // 8) % 4].extend(records[i:i + 8])
    if any(len(g) != 80 for g in groups):
        raise ValueError('Four replicas must retain ten original batch8 cohorts each')
    return groups


def load_groups(plan):
    source = Path(plan['cases_file'])
    if digest(source) != plan['cases_file_sha256']:
        raise ValueError('Source attack bytes changed')
    groups = assignments_for_cases(json.loads(source.read_text()))
    if fingerprint(groups) != plan['assignments_sha256']:
        raise ValueError('Original native batch cohort assignments changed')
    return groups


def worker(run, index):
    plan = json.loads((run / 'plan.json').read_text())
    if digest(ROOT / 'scripts/probe_gcg_completion_replicas.py') != plan['native_worker_sha256']:
        raise ValueError('Native worker source changed after this trial was bound')
    groups = load_groups(plan)
    native_probe.prompt_groups = lambda: [[r['test_case'] for r in group] for group in groups]
    native_probe.worker(run, index)


def run_native_pool(run, *, runtime, models_file, cases_file, identity, reference=None):
    """Caller owns the GPU lease; preserve native batch membership and HF calls."""
    run.mkdir(parents=True, exist_ok=True)
    groups = assignments_for_cases(json.loads(cases_file.read_text()))
    plan = {'task': 'native-NF4-GCG-four-replica-generation-v1', 'model_identity': identity,
        'runtime': str(runtime), 'native_source_sha256': digest(runtime / 'generate_completions.py'),
        'models_file': str(models_file), 'models_file_sha256': digest(models_file),
        'cases_file': str(cases_file), 'cases_file_sha256': digest(cases_file),
        'assignments_sha256': fingerprint(groups), 'native_worker_sha256': digest(native_probe.__file__),
        'pool_source_sha256': digest(__file__), 'replicas': 4, 'batch_per_replica': 8,
        'max_new_tokens': 512, 'retain_answers': True, 'API_calls': 0,
        'production_enablement': False, 'reference_completions': str(reference) if reference else None,
        'reference_completions_sha256': digest(reference) if reference else None,
        'scope': 'Inference only on existing GCG cases, no attacks searched or training examples created'}
    if identity['precision'] != 'nf4':
        raise ValueError('NF4 baseline precision must be preserved')
    bind(run / 'identity', plan)
    if list(run.glob('replica*.ready.json')) or (run / 'parallel_start.json').exists():
        raise ValueError('Owned worker state already exists; inspect instead of restarting')
    write_json(run / 'plan.json', plan)
    def publish(state, **extra):
        write_json(run / 'status.json', {'state': state, 'pid': os.getpid(),
            'observed_epoch': time.time(), 'production_enablement': False, 'API_calls': 0, **extra})
    children = []
    try:
        for index in range(4):
            with (run / f'replica{index}.log').open('a') as output:
                child = subprocess.Popen([str(ROOT / '.venv-harmbench/bin/python'), '-u',
                    str(Path(__file__).absolute()), '--worker', str(index), '--run', str(run)],
                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=dict(os.environ,
                        CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'))
            children.append(child)
        began_loading = time.monotonic()
        while not all((run / f'replica{i}.ready.json').exists() for i in range(4)):
            if any(p.poll() is not None for p in children):
                raise RuntimeError('A replica exited before the generation start barrier')
            if time.monotonic() - began_loading > 900:
                raise RuntimeError('Owned replicas did not load within15 minutes')
            publish('loading_owned_replicas', child_pids=[p.pid for p in children])
            time.sleep(2)
        started = time.monotonic()
        write_json(run / 'parallel_start.json', {'start_epoch': time.time(), 'replicas': 4})
        publish('generating_native_answers_four_replicas', child_pids=[p.pid for p in children])
        samples = []
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('A native replica failed; preserve all cases and logs')
            memory, utilization = [int(s.strip()) for s in subprocess.check_output(['nvidia-smi',
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(',')]
            samples.append({'elapsed': time.monotonic() - started,
                'memory_MiB': memory, 'utilization_percent': utilization})
            if memory > 76 * 1024 or time.monotonic() - started > 3600:
                raise RuntimeError('Owned inference exceeded memory headroom or one-hour timing guard')
            time.sleep(2)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError('A native replica did not complete normally')
        outputs = {}
        results = []
        for index, group in enumerate(groups):
            answers = json.loads((run / f'replica{index}.group{index}.answers.json').read_text())
            if (answers['input_sha256'] != fingerprint([r['test_case'] for r in group])
                    or len(answers['answers']) != len(group)):
                raise ValueError('Owned replica output has a different input cohort or missing answers')
            result = json.loads((run / f'replica{index}.result.json').read_text())
            measured = result['measurements'][0]
            if measured['answers_sha256'] != fingerprint(answers['answers']):
                raise ValueError('Owned replica answer checksum differs')
            results.append(result)
            for row, answer in zip(group, answers['answers']):
                outputs[row['behavior_id']] = [{'test_case': row['test_case'], 'generation': answer}]
        ordered = {key: outputs[key] for key in json.loads(cases_file.read_text())}
        with (run / 'completions.json').open('w') as output:
            json.dump(ordered, output, indent=4)
        if digest(cases_file) != plan['cases_file_sha256']:
            raise ValueError('Original source attack file changed during generation')
        parity = None
        equal = []
        if reference:
            if digest(reference) != plan['reference_completions_sha256']:
                raise ValueError('Saved baseline reference changed during the trial')
            expected = json.loads(reference.read_text())
            if set(expected) != set(ordered):
                raise ValueError('Saved baseline has different coverage')
            equal = [key for key in ordered if expected[key] == ordered[key]]
            parity = len(equal) == 320
        result = {'state': 'completed', 'behaviors': 320,
            'generation_seconds': max(r['generation_seconds'] for r in results),
            'max_GPU_memory_MiB': max(s['memory_MiB'] for s in samples),
            'all320_case_and_answer_records_identical_to_reference': parity,
            'records_identical_to_reference': len(equal) if reference else None,
            'whole_completion_file_byte_identical_to_reference': digest(run / 'completions.json') == digest(reference) if reference else None,
            'completions_sha256': digest(run / 'completions.json'), 'model_identity': identity,
            'replicas': 4, 'batch_each': 8, 'max_new_tokens': 512, 'GPU_samples': samples,
            'source_plan_sha256': digest(run / 'plan.json'), 'production_enablement': False,
            'API_calls': 0, 'scope': plan['scope'], 'does_not_prove_other_checkpoints_identical': True}
        write_json(run / 'result.json', result)
        publish('completed', identical_records=len(equal) if reference else None, full_reference_parity=parity)
        return result
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/security_hardening_20261109_v1/GCG_completion_replication_full320')
    args = parser.parse_args()
    run = args.run.absolute()
    if args.worker is not None:
        worker(run, args.worker)
        return
    import fcntl
    from fineval_pipeline.gpu_lease import acquire_a100_lease
    run.mkdir(parents=True, exist_ok=True)
    singleton = (run / '.supervisor.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    probe = json.loads((ROOT / 'outputs/security_hardening_20261109_v1/GCG_completion_replica_probe/result.json').read_text())
    if (not probe['all_answer_and_generated_token_ids_identical'] or probe['speedup'] < 1.5
            or probe['max_GPU_memory_MiB'] > 70 * 1024):
        raise ValueError('Inspect the completed benign parallel probe before a full320 trial')
    baseline = ROOT / 'outputs/security_hardening_20261109_v1/formal_finance_scoring'
    if json.loads((baseline / 'status.json').read_text())['state'] != 'completed':
        raise ValueError('Never interrupt the original financial GCG scoring')
    folder = baseline / 'evaluation/M1/harmbench/GCG'
    summary = json.loads((folder / 'summary.json').read_text())
    if digest(folder / 'completions.json') != summary['judge_protocol']['completions_sha256']:
        raise ValueError('Officially scored baseline answers changed')
    lease = acquire_a100_lease('full320 native NF4 replica inference equivalence trial', run / 'gpu_lease.json')
    try:
        run_native_pool(run, runtime=ROOT / 'runtime/HarmBench_stage2_nf4_gcg_optimized_v2',
            models_file=folder / 'models.yaml', cases_file=folder / 'test_cases/test_cases.json',
            identity=summary['attack']['model_identity'], reference=folder / 'completions.json')
    finally:
        lease.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--worker' not in sys.argv:
            target = Path(sys.argv[sys.argv.index('--run') + 1]) if '--run' in sys.argv else ROOT / 'outputs/security_hardening_20261109_v1/GCG_completion_replication_full320'
            write_json(target / 'status.json', {'state': 'failed', 'pid': os.getpid(),
                'error_type': type(error).__name__, 'reason': str(error), 'production_enablement': False, 'API_calls': 0})
        raise
