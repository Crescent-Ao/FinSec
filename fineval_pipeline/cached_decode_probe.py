"""Experimental inference-only cached-convolution adapter, disabled by default.

This module does not change the canonical HarmBench runtime or any live worker.
Only a dedicated probe installs it temporarily and restores the original callable.
"""


def fla_cached_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    import torch
    from fla.modules.causal_conv1d.ops import causal_conv1d_update
    if torch.is_grad_enabled():
        raise RuntimeError('Experimental cached decode adapter is inference-only')
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != 1:
        raise ValueError('Experimental cached decode only accepts one new token per sequence')
    if conv_state.shape[:2] != hidden_states.shape[:2] or conv_state.shape[-1] != weight.shape[-1]:
        raise ValueError('Cached convolution shape differs from its declared kernel width')
    output, _ = causal_conv1d_update(hidden_states.to(weight.dtype).transpose(1, 2),
        cache=conv_state, weight=weight, bias=bias, activation=activation)
    return output.transpose(1, 2).to(hidden_states.dtype)
