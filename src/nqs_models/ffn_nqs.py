"""FFNN-based Neural Quantum State (NQS) models.

This file defines FFNN-based NQS models:
1. FFNNNQS: Real-valued log-amplitude (baseline)
2. ComplexFFNNNQS: Complex-valued with separate amplitude and phase networks

The complex model is designed to solve the "sign problem" where real NQS cannot
represent wavefunctions with mixed signs (e.g., H₂ ground state has both positive
and negative coefficients).
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


class ComplexFFNNNQS(nn.Module):
    """Complex-valued FFNN NQS with separate amplitude and phase networks.

    This model represents the wavefunction as:
        ψ(x) = exp(log_amp(x)) * exp(i * phase(x))
             = |ψ(x)| * exp(i * φ(x))

    where:
        - log_amp(x) is the log-amplitude (real-valued)
        - phase(x) is the phase φ ∈ [0, 2π) (real-valued, interpreted as angle)

    This allows the wavefunction to have both positive and negative (and complex)
    coefficients, solving the "sign problem" that limits real-valued NQS.

    For H₂ ground state: |ψ_GS⟩ ≈ 0.99|1100⟩ - 0.14|0011⟩
    The negative sign is represented by φ(0011) = π while φ(1100) = 0.

    Training uses the VMC loss with local energy:
        E_loc(x) = <x|H|ψ> / <x|ψ>

    For complex wavefunctions, the ratio ψ(x')/ψ(x) involves both amplitude
    and phase differences.
    """

    def __init__(self, config: FFNNNQSConfig) -> None:
        super().__init__()
        self.config = config

        # Amplitude network: outputs log|ψ(x)|
        amp_layers = []
        in_dim = config.n_visible
        for _ in range(config.n_layers):
            amp_layers.append(nn.Linear(in_dim, config.n_hidden))
            amp_layers.append(nn.Tanh())  # Tanh for smoother gradients
            in_dim = config.n_hidden
        amp_layers.append(nn.Linear(in_dim, 1))
        self.amplitude_net = nn.Sequential(*amp_layers)

        # Phase network: outputs φ(x) ∈ [0, 2π)
        # Using separate network for phase to decouple amplitude and phase learning
        phase_layers = []
        in_dim = config.n_visible
        for _ in range(config.n_layers):
            phase_layers.append(nn.Linear(in_dim, config.n_hidden))
            phase_layers.append(nn.Tanh())
            in_dim = config.n_hidden
        phase_layers.append(nn.Linear(in_dim, 1))
        self.phase_net = nn.Sequential(*phase_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-amplitude and phase for a batch of bitstrings.

        Parameters
        ----------
        x:
            Tensor of shape (batch, n_visible) containing 0/1 values.

        Returns
        -------
        log_amp:
            Tensor of shape (batch,) with real-valued log-amplitudes log|ψ(x)|.
        phase:
            Tensor of shape (batch,) with phase values φ(x) ∈ [0, 2π).
        """
        x = x.float()
        log_amp = self.amplitude_net(x).squeeze(-1)
        # Use 2π * sigmoid to constrain phase to [0, 2π)
        phase = 2 * torch.pi * torch.sigmoid(self.phase_net(x).squeeze(-1))
        return log_amp, phase

    def log_psi_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Return complex log-wavefunction as complex tensor.

        log ψ(x) = log|ψ(x)| + i * φ(x)

        Parameters
        ----------
        x:
            Tensor of shape (batch, n_visible).

        Returns
        -------
        log_psi:
            Complex tensor of shape (batch,) with log ψ(x).
        """
        log_amp, phase = self.forward(x)
        return torch.complex(log_amp, phase)

    def psi_ratio(
        self, x_prime: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute wavefunction ratio ψ(x')/ψ(x) as complex tensor.

        This is needed for computing local energy E_loc(x) = Σ_x' H_{x,x'} * ψ(x')/ψ(x).

        Parameters
        ----------
        x_prime:
            New configurations, shape (batch, n_visible) or (batch, n_connected, n_visible).
        x:
            Reference configurations, shape (batch, n_visible).

        Returns
        -------
        ratio:
            Complex tensor of shape (batch,) or (batch, n_connected) with ψ(x')/ψ(x).
        """
        log_amp_prime, phase_prime = self.forward(x_prime)
        log_amp, phase = self.forward(x)

        # Handle broadcasting for connected configurations
        if x_prime.dim() == 3:
            # x_prime: (batch, n_connected, n_visible)
            # Need to expand x's values
            log_amp = log_amp.unsqueeze(1)  # (batch, 1)
            phase = phase.unsqueeze(1)  # (batch, 1)

        # ψ(x')/ψ(x) = exp(log_amp' - log_amp) * exp(i * (phase' - phase))
        log_ratio_real = log_amp_prime - log_amp
        phase_diff = phase_prime - phase

        # Return as complex: exp(log_ratio_real + i * phase_diff)
        return torch.exp(torch.complex(log_ratio_real, phase_diff))

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probability log p(x) ∝ 2 * log|ψ(x)| (unnormalized).

        Note: Sampling only depends on |ψ|², so phase is not needed here.
        """
        log_amp, _ = self.forward(x)
        return 2.0 * log_amp

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device | None = None,
        n_warmup: int = 100,
        n_thin: int = 10,
    ) -> torch.Tensor:
        """Metropolis-Hastings MCMC sampler (same as real NQS, uses |ψ|² only)."""
        device = device or next(self.parameters()).device
        n_visible = self.config.n_visible

        states = torch.randint(0, 2, (n_samples, n_visible), device=device)
        log_probs = self.log_prob(states)

        for _ in range(n_warmup):
            states, log_probs = self._mcmc_step(states, log_probs)

        samples = []
        for _ in range(1):
            for _ in range(n_thin):
                states, log_probs = self._mcmc_step(states, log_probs)
            samples.append(states.clone())

        return samples[0]

    def _mcmc_step(
        self, states: torch.Tensor, log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one Metropolis-Hastings step (same as real NQS)."""
        n_chains = states.shape[0]
        n_visible = states.shape[1]

        flip_indices = torch.randint(0, n_visible, (n_chains,), device=states.device)
        proposed_states = states.clone()
        proposed_states[torch.arange(n_chains, device=states.device), flip_indices] = (
            1 - proposed_states[torch.arange(n_chains, device=states.device), flip_indices]
        )

        proposed_log_probs = self.log_prob(proposed_states)
        log_acceptance = proposed_log_probs - log_probs

        accept = torch.log(torch.rand(n_chains, device=states.device)) < log_acceptance

        new_states = torch.where(accept.unsqueeze(-1), proposed_states, states)
        new_log_probs = torch.where(accept, proposed_log_probs, log_probs)

        return new_states, new_log_probs
