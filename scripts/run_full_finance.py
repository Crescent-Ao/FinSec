"""Durable single-GPU job: full audit -> one epoch -> FinEval -> MIA -> Agent.

The detached driver records failures rather than treating partial output as a
completed experiment. Rerunning the same command resumes a verified checkpoint.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.training_options import read_parameters
PYTHON = ROOT/'.venv-eval/bin/python'


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); temporary.replace(path)


def storage_check(run):
    available = shutil.disk_usage(run).free
    if available < 40*1024**3: raise RuntimeError('Less than 40 GiB available on experiment filesystem')
    path = run/'storage_probe.tmp'; content = os.urandom(1024*1024)
    with path.open('wb') as f: f.write(content); f.flush(); os.fsync(f.fileno())
    if path.read_bytes() != content: raise RuntimeError('Storage write/read verification failed')
    path.unlink()
    return {'free_GiB': available/1024**3, 'write_read_fsync': 'passed', 'path': str(run)}


def latest_checkpoint(run):
    checkpoints = sorted((run/'training').glob('**/checkpoint-*/trainer_state.json'), key=lambda p: int(p.parent.name.split('-')[-1]), reverse=True)
    for state in checkpoints:
        needed = ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'args.json')
        if all((state.parent/f).is_file() and (state.parent/f).stat().st_size > 0 for f in needed) and list(state.parent.glob('rng_state*.pth')):
            json.loads(state.read_text())
            return state.parent
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--training-parameters', type=Path)
    parser.add_argument('--result-prefix', default='full_finance')
    parser.add_argument('--acceleration',type=Path)
    args = parser.parse_args(); run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    logs = run/'logs'; logs.mkdir(exist_ok=True)
    # One driver owns a run; a dead driver can be restarted without deleting data.
    import fcntl
    lock = (run/'driver.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = run/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'created_utc': now(), 'stages': {}}
    status['driver_pid'] = os.getpid(); status['run'] = str(run)
    parameter_file = args.training_parameters or (run/'training_parameters.json' if (run/'training_parameters.json').exists() else None)
    parameters = read_parameters(parameter_file)
    acceleration = json.loads(args.acceleration.read_text()) if args.acceleration else None
    status['gpu'] = args.gpu; status['parameters'] = parameters
    status.pop('error', None)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
               HF_DATASETS_CACHE=str(run/'dataset_cache'), TRAIN_BASELINE_ROOT=str(run/'baseline'))
    if parameter_file: env['TRAIN_PARAMETERS'] = str(parameter_file.resolve())
    def update(phase):
        status['phase'] = phase; status['updated_utc'] = now(); save(status_path, status)
    def stage(name, command, extra=None):
        if status['stages'].get(name, {}).get('status') == 'passed': return
        update(name); entry = {'status': 'running', 'started_utc': now(), 'command': [str(s) for s in command]}
        status['stages'][name] = entry; save(status_path, status)
        print('START', name, now(), flush=True)
        with (logs/(name+'.log')).open('a') as output:
            child = subprocess.Popen([str(s) for s in command], cwd=ROOT, env=env | (extra or {}), stdout=output, stderr=subprocess.STDOUT)
            entry['pid'] = child.pid; save(status_path, status)
            while child.poll() is None:
                time.sleep(15); update(name)
            code = child.returncode
        entry.update(returncode=code, ended_utc=now(), status='passed' if code == 0 else 'failed'); save(status_path, status)
        if code: raise RuntimeError(name+' exited '+str(code)+'; inspect '+str(logs/(name+'.log')))
    try:
        update('preflight'); status['storage'] = storage_check(run)
        if not (run/'plan.json').exists():
            save(run/'plan.json', {'scope': 'one pass over all eligible financial training data; no row cap', 'gpu': args.gpu,
                 'base': str(ROOT/'models/Qwen3.5-9B'), 'epochs': 1, 'quantization': 'NF4 QLoRA, BF16 computation',
                 'parameters': parameters, 'max_length': 4096, 'seed': 42,
                 'source_pools': {'fin-glaive': 645232, 'fin-cfa-graphgen': 785149},
                 'post_training_tasks': ['full FinEval Academic val', 'fixed stratified actual-training MIA M0/M1', 'M1 Agent diagnostic fixtures'],
                 'pending_competition_stages': ['Full GCG/PAIR safety assessment and actual M2 safety training are separate from this financial SFT run.'],
                 'storage_reason': 'Capacity disk returned Errno 5 during previous pilot; system filesystem verified writable.'})
        baseline = run/'baseline/M0/fineval'
        if not (baseline/'summary.json').exists():
            template = yaml.safe_load((ROOT/'configs/distilled_v1.yaml').read_text())
            template['output_root'] = str(run/'baseline'); template['hardware']['cuda_visible_devices'] = str(args.gpu)
            baseline_cfg = run/'baseline.yaml'; baseline_cfg.write_text(yaml.safe_dump(template, sort_keys=False))
            stage('baseline', [PYTHON, '-m', 'fineval_pipeline', '--config', baseline_cfg, 'run', '--models', 'M0', '--tasks', 'fineval'])
        base = json.loads((baseline/'summary.json').read_text())
        if base['examples'] != 1151 or base['subjects'] != 34: raise RuntimeError('Incomplete frozen baseline')
        resources = run/'resource_downloads.json'
        if not resources.exists():
            stage('downloads', [PYTHON, ROOT/'scripts/download_finance_resources.py', '--datasets-only',
                               '--destination', ROOT/'data/full_finance_public', '--manifest-output', resources])
        if not (run/'manifest.json').exists():
            stage('preparation', [PYTHON, ROOT/'scripts/prepare_full_finance.py', run, '--workers', args.workers,
                                  '--resource-manifest', resources])
        if not (run/'manifest.json').exists(): raise RuntimeError('Preparation did not freeze a complete manifest')
        digest = sha(run/'manifest.json')
        if status.get('dataset_manifest_sha256') not in (None, digest): raise RuntimeError('Frozen dataset manifest changed')
        status['dataset_manifest_sha256'] = digest
        manifest = json.loads((run/'manifest.json').read_text())
        if manifest.get('data_scale') != 'full_filtered': raise RuntimeError('Refusing a subset manifest')
        status['data'] = {s: {k: e[k] for k in ('rows', 'tokens', 'source_counts')} for s, e in manifest['splits'].items()}
        status['storage_before_training'] = storage_check(run)
        update('waiting_for_gpu')
        while True:
            training_gpus = acceleration['gpus'] if acceleration else [args.gpu]
            used = max(int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=30).strip()) for gpu in training_gpus)
            if used < 1000: break
            status['gpu_memory_used_MiB'] = used; update('waiting_for_gpu'); time.sleep(30)
        if not (run/'memory_probe/audit.json').exists():
            stage('memory_probe', ['bash', ROOT/'scripts/train_distilled_finance.sh', run, 'probe'])
        if not (run/'training/audit.json').exists():
            checkpoint = latest_checkpoint(run)
            extra = {'TRAIN_RESUME_CHECKPOINT': str(checkpoint)} if checkpoint else {}
            command = [PYTHON, ROOT/'scripts/launch_accelerated_training.py',run,args.acceleration] if acceleration else ['bash', ROOT/'scripts/train_distilled_finance.sh', run]
            stage('training', command, extra)
        audit = json.loads((run/'training/audit.json').read_text())
        if not 1 <= audit['epoch'] < 1.0001: raise RuntimeError('Did not complete exactly one epoch')
        stage('fineval', [PYTHON, ROOT/'scripts/complete_distilled_finance.py', run,
                          '--baseline', run/'baseline', '--result-prefix', args.result_prefix, '--gpu', args.gpu])
        cfg = run/'evaluation.yaml'
        stage('privacy', [PYTHON, '-m', 'fineval_pipeline', '--config', cfg, 'run', '--models', 'M0', 'M1', '--tasks', 'privacy'])
        stage('agent_diagnostic', [PYTHON, '-m', 'fineval_pipeline', '--config', cfg, 'run', '--models', 'M1', '--tasks', 'agent'])
        update('completed'); print('COMPLETED', now(), flush=True)
    except BaseException as error:
        status['error'] = str(error); update('failed'); raise


if __name__ == '__main__': main()
