"""Utility functions for NQS models (initialization, diagnostics)."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def init_weights_xavier(modules: Iterable[nn.Module]) -> None:
    """Apply Xavier initialization to Linear layers in the given modules."""
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
