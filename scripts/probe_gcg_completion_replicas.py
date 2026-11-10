"""Measure native NF4 batch8 replication after the queued inference thread probe.

Synthetic benign prompts only. No attack, model, production setting, or saved
score is changed. A successful probe is diagnostic, not automatic deployment.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import bind, digest, fingerprint, write_json

RELATIVE = Path('outputs/security_hardening_20261109_v1/GCG_completion_replica_probe')


def prompt_groups():
    topics = ['bond duration and interest rate sensitivity', 'cash flow and accrual accounting',
        'portfolio diversification and correlation', 'authentication and authorization',
        'authenticated encryption and random nonces', 'password hashing and salt',
        'simple interest and compound interest', 'financial statement reconciliation']
    groups = []
    for group in range(4):
        groups.append([f'For an educational lecture, explain {topic} using fictional examples. '
            'Discuss assumptions, common mistakes, and verification methods in detail. '
            'Write at least1200 words. Background notes: ' +
            ('Examples must use fictional amounts and test credentials. ' * (8 + 18 * group))
            for topic in topics])
    return groups


def worker(run, index):
    import runpy
    import torch
    import yaml
    from fineval_pipeline.harmbench_completions import capped_batch_finder
    plan = json.loads((run / 'plan.json').read_text())
    runtime = Path(plan['runtime'])
    models = Path(plan['models_file'])
    if (digest(runtime / 'generate_completions.py') != plan['native_source_sha256']
            or digest(models) != plan['models_file_sha256']):
        raise ValueError('Pinned native inference source or model configuration changed')
    sys.path.insert(0, str(runtime))
    namespace = runpy.run_path(str(runtime / 'generate_completions.py'), run_name='replica_probe_upstream')
    scope = namespace['load_generation_function'].__globals__
    scope['find_executable_batch_size'] = capped_batch_finder(scope['find_executable_batch_size'], 8)
    model_config = yaml.safe_load(models.read_text())['target']['model']
    generate = namespace['load_generation_function'](model_config, 512,
        test_cases_path=str(run / 'synthetic_unused.json'), generate_with_vllm=False)
    model = generate.keywords['model']
    if not model.is_loaded_in_4bit:
        raise ValueError('The probe requires the original NF4 model')
    groups = prompt_groups()
    selected = list(range(4)) if index == -1 else [index]
    generate.keywords['max_new_tokens'] = 2
    generate(test_cases=groups[selected[0]])
    generate.keywords['max_new_tokens'] = 512
    name = 'serial' if index == -1 else f'replica{index}'
    write_json(run / f'{name}.ready.json', {'pid': os.getpid(), 'CPU_threads': torch.get_num_threads(),
        'loaded_in_4bit': True, 'warmup_completed': True, 'assigned_groups': selected})
    if index != -1:
        while not (run / 'parallel_start.json').exists():
            time.sleep(.1)
    captured = []
    original = model.generate
    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        captured.append(output[:, kwargs['inputs'].shape[1]:].detach().cpu().tolist())
        return output
    model.generate = capture
    measured = []
    torch.cuda.synchronize()
    started = time.monotonic()
    try:
        for group in selected:
            captured.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.monotonic()
            answers = generate(test_cases=groups[group])
            torch.cuda.synchronize()
            if plan.get('retain_answers', False):
                write_json(run / f'{name}.group{group}.answers.json',
                    {'group': group, 'answers': answers, 'input_sha256': fingerprint(groups[group])})
            measured.append({'group': group, 'seconds': time.monotonic() - begin,
                'answers_sha256': fingerprint(answers), 'generated_token_ids_sha256': fingerprint(captured),
                'output_tokens': sum(len(ids) for batch in captured for ids in batch),
                'input_sha256': fingerprint(groups[group]), 'examples': len(answers),
                'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30})
    finally:
        model.generate = original
    write_json(run / f'{name}.result.json', {'state': 'completed', 'pid': os.getpid(),
        'generation_seconds': time.monotonic() - started, 'measurements': measured,
        'production_enablement': False, 'API_calls': 0})


def supervise(run):
    import psutil
    from fineval_pipeline.gpu_lease import acquire_a100_lease
    run.mkdir(parents=True, exist_ok=True)
    singleton = (run / '.probe.lock').open('a')
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
    def publish(state, **kwargs):
        write_json(run / 'status.json', {'state': state, 'pid': os.getpid(),
            'observed_epoch': time.time(), 'production_enablement': False, 'API_calls': 0, **kwargs})
    predecessor = ROOT / 'outputs/security_hardening_20261109_v1/GCG_completion_thread_probe'
    while True:
        state = json.loads((predecessor / 'status.json').read_text())
        if state['state'] == 'completed':
            break
        if state['state'] in ['failed', 'requires_inspection']:
            publish('requires_inspection', reason='Inspect previous inference diagnostic before another GPU probe')
            return
        try:
            proc = psutil.Process(int(state['pid']))
            live = proc.status() != 'zombie' and 'probe_gcg_completion_threads.py' in ' '.join(proc.cmdline())
        except psutil.NoSuchProcess:
            live = False
        if not live:
            publish('requires_inspection', reason='Original predecessor process is missing; no restart')
            return
        publish('waiting_for_original_scoring_and_thread_probe', predecessor_pid=state['pid'],
            predecessor_state=state['state'])
        time.sleep(20)
    lease = acquire_a100_lease('native NF4 GCG batch8 four-replica inference diagnostic', run / 'gpu_lease.json')
    children = []
    try:
        runtime = ROOT / 'runtime/HarmBench_stage2_nf4_gcg_optimized_v2'
        baseline = ROOT / 'outputs/security_hardening_20261109_v1/formal_finance_scoring'
        if json.loads((baseline / 'status.json').read_text())['state'] != 'completed':
            raise ValueError('Preserve incomplete original GCG scoring')
        context = json.loads((baseline / 'evaluation/M1/harmbench/GCG/run.json').read_text())['context']
        models = baseline / 'evaluation/M1/harmbench/GCG/models.yaml'
        plan = {'task': 'native-completion-replica-probe-v1', 'model_identity': context['model'],
            'runtime': str(runtime), 'native_source_sha256': digest(runtime / 'generate_completions.py'),
            'models_file': str(models), 'models_file_sha256': digest(models),
            'probe_source_sha256': digest(__file__), 'groups_sha256': fingerprint(prompt_groups()),
            'replicas': 4, 'batch_per_replica': 8, 'max_new_tokens': 512,
            'production_enablement': False, 'API_calls': 0,
            'scope': '32 fixed synthetic benign prompts; same batches serial vs concurrent; no formal ASR or test inputs'}
        bind(run / 'identity', plan)
        if (run / 'parallel_start.json').exists() or (run / 'serial.result.json').exists():
            raise ValueError('Saved measurements exist; preserve them rather than restart a probe')
        write_json(run / 'plan.json', plan)
        def launch(index):
            name = 'serial' if index == -1 else f'replica{index}'
            with (run / f'{name}.log').open('a') as output:
                proc = subprocess.Popen([str(ROOT / '.venv-harmbench/bin/python'), '-u',
                    str(Path(__file__).absolute()), '--worker', str(index)], cwd=ROOT,
                    stdout=output, stderr=subprocess.STDOUT, env=dict(os.environ,
                        CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'))
            children.append(proc)
            return proc
        serial = launch(-1)
        publish('measuring_serial_reference', child_pid=serial.pid)
        if serial.wait():
            raise RuntimeError('Serial benign reference failed; no automatic restart')
        pool = [launch(i) for i in range(4)]
        started_wait = time.monotonic()
        while not all((run / f'replica{i}.ready.json').exists() for i in range(4)):
            if any(p.poll() is not None for p in pool):
                raise RuntimeError('A replica exited before the start barrier')
            if time.monotonic() - started_wait > 900:
                raise RuntimeError('Owned diagnostic replicas did not load within15 minutes')
            publish('loading_owned_replicas', child_pids=[p.pid for p in pool])
            time.sleep(2)
        parallel_start = time.monotonic()
        write_json(run / 'parallel_start.json', {'start_epoch': time.time(), 'replicas': 4})
        publish('measuring_concurrent_replicas', child_pids=[p.pid for p in pool])
        samples = []
        while any(p.poll() is None for p in pool):
            if any(p.poll() not in (None, 0) for p in pool):
                raise RuntimeError('A diagnostic replica failed; formal evaluation is unchanged')
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip()
            memory, utilization = [int(s.strip()) for s in gpu.split(',')]
            samples.append({'elapsed': time.monotonic() - parallel_start,
                'GPU_memory_MiB': memory, 'GPU_utilization_percent': utilization})
            if memory > 76 * 1024:
                raise RuntimeError('Owned diagnostic exceeded76GiB headroom guard')
            if time.monotonic() - parallel_start > 1800:
                raise RuntimeError('Owned concurrent diagnostic exceeded30 minutes')
            time.sleep(1)
        serial_result = json.loads((run / 'serial.result.json').read_text())
        parallel_results = [json.loads((run / f'replica{i}.result.json').read_text()) for i in range(4)]
        before = serial_result['measurements']
        after = [x['measurements'][0] for x in parallel_results]
        parity = all(all(a[k] == b[k] for k in ['group', 'input_sha256', 'answers_sha256',
            'generated_token_ids_sha256', 'examples', 'output_tokens']) for a, b in zip(before, after))
        # Each worker measures its own synchronized generation span after the
        # same start barrier. Exclude process teardown and file-polling latency.
        concurrent_seconds = max(x['generation_seconds'] for x in parallel_results)
        result = {'state': 'completed', 'serial_generation_seconds': serial_result['generation_seconds'],
            'concurrent_generation_seconds': concurrent_seconds,
            'speedup': serial_result['generation_seconds'] / concurrent_seconds,
            'all_answer_and_generated_token_ids_identical': parity,
            'replicas': 4, 'batch_per_replica': 8, 'max_new_tokens': 512,
            'max_GPU_memory_MiB': max(x['GPU_memory_MiB'] for x in samples) if samples else None,
            'GPU_samples': samples, 'serial': before, 'concurrent': after,
            'production_enablement': False, 'API_calls': 0, 'scope': plan['scope'],
            'limitations': 'Synthetic benign parity and timing do not prove all formal320 outputs equivalent. '
                'Any production adoption requires an explicit matched-generation protocol and complete evidence.'}
        write_json(run / 'result.json', result)
        publish('completed', token_parity=parity, speedup=result['speedup'])
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
        lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=[-1, 0, 1, 2, 3])
    args = parser.parse_args()
    run = ROOT / RELATIVE
    if args.worker is not None:
        worker(run, args.worker)
    else:
        try:
            supervise(run)
        except Exception as error:
            write_json(run / 'status.json', {'state': 'failed', 'pid': os.getpid(),
                'error_type': type(error).__name__, 'reason': str(error),
                'production_enablement': False, 'API_calls': 0})
            raise
