"""Typed SFT/DPO/GRPO entry point used by the stage-2 experiment chain."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swift.cli.utils import try_use_single_device_mode
from swift.arguments import SftArguments, RLHFArguments
from swift.pipelines import sft_main, rlhf_main
from fineval_pipeline.core import write_json


def normalize_config(plan):
    options = plan['arguments']
    if plan['algorithm'] == 'grpo' and options.get('eval_strategy', 'no') != 'no':
        generations = options.get('num_generations_eval') or options['num_generations']
        batch = options.get('per_device_eval_batch_size', 1)
        options['num_generations_eval'] = generations
        if batch % generations:
            options['per_device_eval_batch_size'] = ((batch + generations - 1) // generations) * generations
            plan['eval_batch_adjustment'] = 'Complete GRPO generation groups for validation'
    return plan


if __name__ == '__main__':
    try_use_single_device_mode()
    plan = normalize_config(json.loads(Path(sys.argv[1]).read_text()))
    write_json(Path(sys.argv[1]), plan)
    if plan['algorithm'] == 'sft':
        sft_main(SftArguments(**plan['arguments']))
    else:
        rlhf_main(RLHFArguments(**plan['arguments']))
