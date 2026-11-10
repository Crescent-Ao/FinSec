import copy
from functools import partial
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from scripts.activation_checkpoint_policy import retain_layer_activations


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.dropout = nn.Dropout(.1)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    def forward(self, hidden):
        return hidden + .05*self.dropout(torch.tanh(self.linear(hidden)))


class ActivationPolicyTests(unittest.TestCase):
    def test_retention_preserves_loss_and_gradients_with_dropout(self):
        torch.manual_seed(42)
        baseline = nn.Sequential(*[Qwen3_5DecoderLayer() for _ in range(32)])
        candidate = copy.deepcopy(baseline)
        selected = retain_layer_activations(candidate, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(sum(not layer.gradient_checkpointing for layer in candidate), 3)
        hidden = torch.randn(4, 8)
        losses = []
        for model in (baseline, candidate):
            torch.manual_seed(123)
            loss = model(hidden.clone()).square().mean()
            loss.backward()
            losses.append(loss.item())
        self.assertEqual(losses[0], losses[1])
        for original, changed in zip(baseline.parameters(), candidate.parameters()):
            torch.testing.assert_close(original.grad, changed.grad, rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            retain_layer_activations(baseline, 32)


if __name__ == '__main__':
    unittest.main()
