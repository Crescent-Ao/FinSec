"""Audited execution optimizations for the pinned official GCG implementation.

Search, gradients, sampling and loss definitions remain upstream. Only unused
logit positions, tokenization transfers, allocator collection and persistence
are changed. Hybrid attention prefix caching deliberately stays disabled.
"""
from pathlib import Path


def fla_causal_conv1d(hidden_states, weight, bias=None, activation=None, **kwargs):
    """FLA's Ampere-compatible Triton conv, matching the HF channel-first API."""
    from fla.modules.causal_conv1d import causal_conv1d
    # HF forwards model flags such as use_cache here. Its reference convolution
    # ignores these **kwargs; cache state is maintained by the enclosing layer.
    output, _ = causal_conv1d(hidden_states.to(weight.dtype).transpose(1, 2),
        weight=weight, bias=bias, activation=activation, backend='triton')
    return output.transpose(1, 2).to(hidden_states.dtype)


def enable_fla_convolution():
    from transformers.models.qwen3_5 import modeling_qwen3_5
    modeling_qwen3_5.causal_conv1d_fn = fla_causal_conv1d


def validate_fla_convolution():
    """Check values and GCG input gradients against HF's reference expression."""
    import torch
    import torch.nn.functional as F
    records = []
    for length in (16, 65, 257):
        torch.manual_seed(length)
        x = torch.randn(2, 128, length, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(128, 4, device='cuda', dtype=torch.bfloat16) * .1
        for activation in (None, 'silu'):
            results = []
            for fused in (False, True):
                z = x.detach().clone().requires_grad_()
                if fused:
                    y = fla_causal_conv1d(z, weight, activation=activation)
                else:
                    y = F.conv1d(z, weight.unsqueeze(1), padding=3, groups=128)[:, :, :length]
                    if activation: y = F.silu(y)
                grad, = torch.autograd.grad(y.float().square().mean(), z)
                results.append((y.detach().float(), grad.detach().float()))
            a, b = results
            value_error = float((a[0]-b[0]).norm()/a[0].norm().clamp_min(1e-12))
            cosine = float(F.cosine_similarity(a[1].flatten(), b[1].flatten(), dim=0))
            row = {'length':length, 'activation':activation, 'output_relative_l2':value_error,
                   'input_gradient_cosine':cosine,
                   'passed':value_error < .02 and cosine > .995 and bool(torch.isfinite(b[1]).all())}
            records.append(row)
    if not all(r['passed'] for r in records):
        raise RuntimeError(f'FLA convolution validation failed: {records}')
    return records


def optimize_gcg(source):
    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError(f'Pinned GCG optimization no longer applies: {old[:90]}')
        source = source.replace(old, new)

    # Qwen3.5 accepts logits_to_keep. Keep the preceding predictor plus all
    # target positions; the final target logit is excluded by the original loss.
    replace('outputs = model(inputs_embeds=input_embeds)',
            'outputs = model(inputs_embeds=input_embeds, logits_to_keep=target_ids.shape[1] + 1)')
    replace('tmp = input_embeds.shape[1] - target_embeds.shape[1]',
            'tmp = logits.shape[1] - target_embeds.shape[1]')
    replace('outputs = self.model(inputs_embeds=input_embeds_batch)',
            'outputs = self.model(inputs_embeds=input_embeds_batch, logits_to_keep=target_ids.shape[1] + 1)')
    replace('tmp = input_embeds_batch.shape[1] - target_ids.shape[1]',
            'tmp = logits.shape[1] - target_ids.shape[1]')
    # Filtering leaves a variable final minibatch. Pad only that minibatch and
    # discard duplicate losses, keeping Triton shapes stable across iterations.
    replace('                input_embeds_batch = input_embeds[i:i+search_batch_size]',
            '                input_embeds_batch = input_embeds[i:i+search_batch_size]\n'
            '                candidate_count = input_embeds_batch.shape[0]\n'
            '                if candidate_count < search_batch_size:\n'
            '                    input_embeds_batch = torch.cat([input_embeds_batch,\n'
            '                        input_embeds_batch[-1:].expand(search_batch_size-candidate_count, -1, -1)], dim=0)')
    replace('            all_loss.append(loss)', '            all_loss.append(loss[:candidate_count])')
    replace('        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))',
            '        if not hasattr(self, "_gcg_vocab_embeds"):\n'
            '            self._gcg_vocab_embeds = embed_layer(torch.arange(0, vocab_size, device=model.device))\n'
            '        vocab_embeds = self._gcg_vocab_embeds')
    replace('        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)\n'
            '        # remove redundant tokens\n'
            '        not_allowed_tokens = torch.unique(not_allowed_tokens)',
            '        if not hasattr(self, "_gcg_nonascii_tokens"):\n'
            '            self._gcg_nonascii_tokens = torch.unique(get_nonascii_toks(tokenizer)).to(device)\n'
            '        not_allowed_tokens = None if allow_non_ascii else self._gcg_nonascii_tokens')

    old = '''            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])
            
            if len(new_sampled_top_indices) == 0:
                print('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)
'''
    new = '''            sampled_cpu = sampled_top_indices.detach().cpu()
            sampled_top_indices_text = tokenizer.batch_decode(sampled_cpu)
            retokenized = tokenizer(sampled_top_indices_text, add_special_tokens=False)['input_ids']
            kept = [j for j, (original, encoded) in enumerate(zip(sampled_cpu.tolist(), retokenized))
                    if original == encoded]
            count = len(sampled_top_indices_text) - len(kept)
            if not kept:
                print('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = sampled_top_indices.index_select(
                    0, torch.tensor(kept, device=device, dtype=torch.long))
'''
    replace(old, new)
    # Keep OOM-retry cleanup in compute_candidates_loss. Normal steps reuse the
    # allocator; releasing live tensors is sufficient and avoids CPU/GPU stalls.
    replace('            del input_embeds, sampled_top_embeds, logits, shift_logits, loss\n'
            '            torch.cuda.empty_cache()\n            gc.collect()',
            '            del input_embeds, sampled_top_embeds, logits, shift_logits, loss, outputs')
    replace('            del outputs, logits, loss\n'
            '            torch.cuda.empty_cache()\n            gc.collect()',
            '            del outputs, logits, shift_logits, loss')
    return source


def checkpointed_generation(method, behaviors, save_dir, method_config, run_id, verbose=False):
    """One durable official output per behavior, deterministic across resumes."""
    import hashlib
    import json
    import os
    import random
    import tempfile
    import time

    import numpy as np
    import torch
    from tqdm import tqdm

    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    completed = len(list((root / 'test_cases_individual_behaviors').glob('*/test_cases.json')))
    total = completed + len(behaviors)
    base_seed = int(os.environ.get('HB_SEED', '42'))
    for behavior in tqdm(behaviors, total=len(behaviors), desc='GCG behaviors (durable)'):
        behavior_id = behavior['BehaviorID']
        seed = int.from_bytes(hashlib.sha256(f'{base_seed}:{behavior_id}'.encode()).digest()[:4], 'big')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        started = time.monotonic()
        test_cases, logs = method.generate_test_cases([behavior], verbose=verbose)
        # Commit test_cases last: upstream skips only this file on resume.
        with tempfile.TemporaryDirectory(prefix='.pending-', dir=root) as temporary:
            method.save_test_cases(temporary, test_cases, logs, method_config=method_config, run_id=run_id)
            pending = Path(method.get_output_file_path(temporary, behavior_id, 'test_cases', run_id)).parent
            destination = Path(method.get_output_file_path(save_dir, behavior_id, 'test_cases', run_id)).parent
            destination.mkdir(parents=True, exist_ok=True)
            files = sorted(pending.iterdir(), key=lambda p: p.name.startswith('test_cases'))
            for path in files:
                os.replace(path, destination / path.name)
        completed += 1
        progress = {'completed': completed, 'total': total, 'last_behavior_id': behavior_id,
                    'behavior_seconds': time.monotonic() - started, 'behavior_seed': seed,
                    'seed_protocol': 'sha256(base_seed:BehaviorID) first 32 bits',
                    'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'cuda_reserved_gib': torch.cuda.memory_reserved() / 2**30}
        temporary = root / 'generation_progress.json.tmp'
        temporary.write_text(json.dumps(progress, indent=2) + '\n')
        os.replace(temporary, root / 'generation_progress.json')
        print('GCG_PROGRESS ' + json.dumps(progress), flush=True)
    # Results already saved; let the official driver finish and merge normally.
    return {}, {}
