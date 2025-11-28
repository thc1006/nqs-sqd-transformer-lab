"""FFNN-based Neural Quantum State (NQS) models.

This file defines a small FFNN-based NQS that outputs a real-valued log-amplitude.
It is intended as a strong yet simple baseline for small systems (e.g. H₂ with
12–14 visible units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class FFNNNQSConfig:
    """Configuration for FFNN-based NQS.

    Attributes
    ----------
    n_visible:
        Number of visible units (e.g. spin orbitals / qubits).
    n_hidden:
        Hidden layer width.
    n_layers:
        Number of hidden layers.
    """

    n_visible: int
    n_hidden: int = 64
    n_layers: int = 2


class FFNNNQS(nn.Module):
    """Simple feed-forward NQS that outputs a scalar log-amplitude.

    For now this is a *real-valued* log ψ model. You can extend this to
    complex-valued representations later if needed.
    """

    def __init__(self, config: FFNNNQSConfig) -> None:
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.n_visible
        for _ in range(config.n_layers):
            layers.append(nn.Linear(in_dim, config.n_hidden))
            layers.append(nn.ReLU())
            in_dim = config.n_hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-amplitude log|ψ(x)| for a batch of bitstrings.

        Parameters
        ----------
        x:
            Tensor of shape (batch, n_visible) containing 0/1 values.

        Returns
        -------
        log_psi:
            Tensor of shape (batch,) with real-valued log-amplitudes.
        """
        x = x.float()
        out = self.net(x)
        return out.squeeze(-1)

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probability log p(x) ∝ 2 * log|ψ(x)| (unnormalized)."""
        return 2.0 * self.forward(x)

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device | None = None) -> torch.Tensor:
        """Very crude independent Bernoulli sampler (placeholder).

        This is **not** a proper MCMC sampler yet. It just samples each visible
        unit independently with p=0.5. You should replace this with Metropolis
        or an autoregressive sampler.

        Returns
        -------
        samples:
            Tensor of shape (n_samples, n_visible) with 0/1 entries.
        """
        device = device or next(self.parameters()).device
        return torch.bernoulli(0.5 * torch.ones(n_samples, self.config.n_visible, device=device))
