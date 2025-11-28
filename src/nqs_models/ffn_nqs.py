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
    def sample(
        self,
        n_samples: int,
        device: torch.device | None = None,
        n_warmup: int = 100,
        n_thin: int = 10,
    ) -> torch.Tensor:
        """Metropolis-Hastings MCMC sampler with single spin-flip proposals.

        This implements a vectorized MCMC sampler that runs n_samples independent
        chains in parallel on the GPU. Each chain:
          1. Starts from a random initial configuration
          2. Performs n_warmup warmup steps (discarded)
          3. Performs n_thin steps per collected sample

        The proposal is a single spin-flip at a random position.
        Acceptance probability: A(x → x') = min(1, |ψ(x')|² / |ψ(x)|²)

        Parameters
        ----------
        n_samples:
            Number of independent samples to generate.
        device:
            Device to run sampling on (default: model's device).
        n_warmup:
            Number of warmup steps to discard per chain (default: 100).
        n_thin:
            Thinning interval - take one sample every n_thin steps (default: 10).

        Returns
        -------
        samples:
            Tensor of shape (n_samples, n_visible) with 0/1 entries.
        """
        device = device or next(self.parameters()).device
        n_visible = self.config.n_visible

        # Initialize n_samples independent chains with random configurations
        states = torch.randint(0, 2, (n_samples, n_visible), device=device)
        log_probs = self.log_prob(states)

        # Warmup phase
        for _ in range(n_warmup):
            states, log_probs = self._mcmc_step(states, log_probs)

        # Sampling phase: collect one sample every n_thin steps
        samples = []
        for _ in range(1):  # We collect exactly n_samples, one per chain
            for _ in range(n_thin):
                states, log_probs = self._mcmc_step(states, log_probs)
            samples.append(states.clone())

        return samples[0]  # Shape: (n_samples, n_visible)

    def _mcmc_step(
        self, states: torch.Tensor, log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one Metropolis-Hastings step with single spin-flip proposal.

        Parameters
        ----------
        states:
            Current states, shape (n_chains, n_visible).
        log_probs:
            Current log probabilities, shape (n_chains,).

        Returns
        -------
        new_states:
            Updated states after acceptance/rejection, shape (n_chains, n_visible).
        new_log_probs:
            Updated log probabilities, shape (n_chains,).
        """
        n_chains = states.shape[0]
        n_visible = states.shape[1]

        # Propose: flip a random spin in each chain
        flip_indices = torch.randint(0, n_visible, (n_chains,), device=states.device)
        proposed_states = states.clone()
        proposed_states[torch.arange(n_chains, device=states.device), flip_indices] = (
            1 - proposed_states[torch.arange(n_chains, device=states.device), flip_indices]
        )

        # Compute acceptance probability
        proposed_log_probs = self.log_prob(proposed_states)
        log_acceptance = proposed_log_probs - log_probs

        # Accept or reject
        accept = torch.log(torch.rand(n_chains, device=states.device)) < log_acceptance

        # Update states and log_probs where accepted
        new_states = torch.where(accept.unsqueeze(-1), proposed_states, states)
        new_log_probs = torch.where(accept, proposed_log_probs, log_probs)

        return new_states, new_log_probs
