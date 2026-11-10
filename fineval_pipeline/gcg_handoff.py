"""Prepare a GRPO checkpoint for independent RTX4090 GCG search."""
import json
import os
from pathlib import Path
import subprocess

from .core import ROOT, digest, fingerprint, load_config, resolve, write_json
from .gcg_shards import protocol
from .harmbench_upstream import prepare


def adapter_files(directory):
    directory = Path(directory)
    files = {p.name: digest(p) for p in sorted(directory.iterdir())
             if p.is_file() and not p.is_symlink()
             and (p.suffix == '.safetensors' or p.name.endswith(('.json', '.jinja')))}
    if not any(name.endswith('.safetensors') for name in files):
        raise ValueError('GRPO checkpoint has no adapter weights')
    return files


def runtime_versions(cfg):
    # The CPU supervisor's environment differs from the GCG execution environment.
    # Keep the venv symlink path: resolving it would select the base interpreter.
    command=[str(resolve(cfg['harmbench']['python'])),'-c',
             'import json;from fineval_pipeline.gcg_shards import versions;print(json.dumps(versions()))']
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''))
    return json.loads(subprocess.check_output(command,cwd=ROOT,env=env,text=True))


def prepare_request(config_path, training_run):
    cfg = load_config(config_path)
    directory = resolve(cfg['output_root']) / 'M2/harmbench/GCG'
    prepare(cfg, 'M2', 'GCG', directory)
    context = json.loads((directory / 'run.json').read_text())['context']
    metadata = json.loads((directory / 'attack_summary.json').read_text())
    expected = protocol(context['model'], metadata, context,
                        digest(resolve(cfg['harmbench']['targets'])), runtime_versions(cfg))
    adapter = resolve(cfg['models']['M2']['adapter'])
    manifest = resolve(cfg['models']['M2']['safety_manifest'])
    request = {'schema': 1, 'training_run': str(training_run),
               'config': str(Path(config_path).absolute()), 'config_sha256': digest(config_path),
               'attack_directory': str(directory), 'output_root': str(resolve(cfg['output_root'])),
               'adapter_path': str(adapter), 'adapter_files': adapter_files(adapter),
               'lineage_path': str(manifest), 'lineage_sha256': digest(manifest),
               'protocol': expected, 'model': 'M2',
               'search_host': '40903', 'scoring_host': 'A100',
               'extra_pair_api_calls': False, 'a100_search_fallback': False}
    request['request_id'] = fingerprint(request)
    folder = Path(training_run) / 'distributed_grpo_gcg'
    path = folder / 'request.json'
    if path.exists() and json.loads(path.read_text()) != request:
        raise ValueError('GRPO GCG handoff changed after publication')
    write_json(path, request)
    return request


def check_receipt(request, receipt):
    if receipt['expected'] != request['protocol']:
        raise ValueError('GRPO search receipt refers to a different model or protocol')
    if receipt['state'] == 'running':
        return False
    if receipt['state'] != 'ready_full':
        raise ValueError('Incomplete RTX4090 search; A100 search fallback is disabled')
    if receipt['validated_behaviors'] != len(request['protocol']['selected_ids']):
        raise ValueError('GRPO GCG receipt is missing behaviors')
    return True
