"""Validated numerical SFT options shared by experiment drivers and Bash."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_PARAMETERS = {'learning_rate': 2e-5, 'lora_rank': 16, 'lora_alpha': 32,
                      'lora_dropout': .05, 'warmup_ratio': .05}


def read_parameters(path=None):
    supplied = json.loads(Path(path).read_text()) if path else {}
    if set(supplied)-set(DEFAULT_PARAMETERS):
        raise ValueError('Unknown training parameter')
    result = DEFAULT_PARAMETERS | supplied
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Nonfinite or nonnumeric parameter: '+key)
    if not 1e-7 <= result['learning_rate'] <= 5e-4: raise ValueError('Learning rate outside experiment range')
    for key, upper in [('lora_rank', 128), ('lora_alpha', 512)]:
        if not isinstance(result[key], int) or not 1 <= result[key] <= upper: raise ValueError('Invalid '+key)
    if not 0 <= result['lora_dropout'] <= .5 or not 0 <= result['warmup_ratio'] <= .2:
        raise ValueError('Invalid dropout or warmup')
    return result


def main():
    run = Path(sys.argv[1]); configured = os.environ.get('TRAIN_PARAMETERS')
    path = Path(configured) if configured else run/'training_parameters.json'
    path = path if path.exists() else None
    if configured and path is None: raise ValueError('Configured parameter file is missing')
    params = read_parameters(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    manifest = json.loads((run/'manifest.json').read_text())
    expected = manifest.get('experiment', {}).get('parameter_sha256')
    if expected and expected != sha: raise ValueError('Frozen experiment parameters changed')
    (run/'training_hyperparameters.json').write_text(json.dumps({'parameters': params, 'parameter_file_sha256': sha}, indent=2))
    for key, value in params.items():
        print('--'+key); print(value)


if __name__ == '__main__': main()
