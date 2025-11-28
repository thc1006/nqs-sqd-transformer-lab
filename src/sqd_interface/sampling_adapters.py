"""Adapters that connect samplers (NQS or baselines) to the SQD interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from src.nqs_models.ffn_nqs import FFNNNQS, ComplexFFNNNQS
from src.nqs_models.transformer_nqs import TransformerNQS


class Sampler(Protocol):
    """Abstract sampler protocol returning bitstring arrays."""

    def sample(self, n_samples: int) -> np.ndarray:
        ...


@dataclass
class BaselineBernoulliSampler:
    """Baseline sampler that draws i.i.d. Bernoulli(0.5) bitstrings."""

    n_visible: int

    def sample(self, n_samples: int) -> np.ndarray:
        bits = np.random.randint(0, 2, size=(n_samples, self.n_visible), dtype=np.int8)
        return bits


@dataclass
class FFNNSampler:
    """Adapter that uses an FFNNNQS model as a sampler.

    Includes optional postselection to enforce particle number conservation,
    which is required for SQD to work properly.
    """

    model: FFNNNQS
    device: torch.device
    n_electrons: int | None = None  # If set, postselect to this particle number
    n_alpha: int | None = None  # If set, also enforce alpha electron count
    n_beta: int | None = None   # If set, also enforce beta electron count

    def sample(self, n_samples: int, oversample_factor: int = 50) -> np.ndarray:
        """Sample bitstrings from the model.

        Parameters
        ----------
        n_samples:
            Number of samples to return.
        oversample_factor:
            When postselecting, oversample by this factor to ensure enough valid samples.

        Returns
        -------
        samples:
            Array of shape (n_samples, n_visible) with 0/1 entries.
        """
        if self.n_electrons is None:
            # No postselection, return raw samples
            samples = self.model.sample(n_samples=n_samples, device=self.device)
            return samples.cpu().numpy().astype("int8")

        # Default alpha/beta split: assume equal
        n_alpha = self.n_alpha if self.n_alpha is not None else self.n_electrons // 2
        n_beta = self.n_beta if self.n_beta is not None else self.n_electrons // 2

        # Postselect to correct particle number AND alpha/beta split
        total_needed = n_samples * oversample_factor
        all_samples = self.model.sample(n_samples=total_needed, device=self.device)
        all_samples_np = all_samples.cpu().numpy()

        # Spin-orbital basis: even indices = alpha, odd indices = beta
        n_visible = all_samples_np.shape[1]
        alpha_indices = np.arange(0, n_visible, 2)
        beta_indices = np.arange(1, n_visible, 2)

        alpha_counts = all_samples_np[:, alpha_indices].sum(axis=1)
        beta_counts = all_samples_np[:, beta_indices].sum(axis=1)

        # Filter to correct alpha AND beta counts
        valid_mask = (alpha_counts == n_alpha) & (beta_counts == n_beta)
        valid_samples = all_samples_np[valid_mask]

        if len(valid_samples) >= n_samples:
            return valid_samples[:n_samples].astype("int8")
        else:
            # Not enough valid samples, return what we have (may cause SQD errors)
            return valid_samples.astype("int8") if len(valid_samples) > 0 else all_samples_np[:n_samples].astype("int8")


@dataclass
class TransformerSampler:
    """Adapter that uses a TransformerNQS model as an autoregressive sampler.

    Includes optional postselection to enforce particle number conservation.
    """

    model: TransformerNQS
    device: torch.device
    n_electrons: int | None = None  # If set, postselect to this particle number
    n_alpha: int | None = None  # If set, also enforce alpha electron count
    n_beta: int | None = None   # If set, also enforce beta electron count

    def sample(self, n_samples: int, oversample_factor: int = 50) -> np.ndarray:
        """Sample bitstrings from the model.

        Parameters
        ----------
        n_samples:
            Number of samples to return.
        oversample_factor:
            When postselecting, oversample by this factor to ensure enough valid samples.

        Returns
        -------
        samples:
            Array of shape (n_samples, n_visible) with 0/1 entries.
        """
        if self.n_electrons is None:
            samples = self.model.sample(n_samples=n_samples, device=self.device)
            return samples.cpu().numpy().astype("int8")

        # Default alpha/beta split: assume equal
        n_alpha = self.n_alpha if self.n_alpha is not None else self.n_electrons // 2
        n_beta = self.n_beta if self.n_beta is not None else self.n_electrons // 2

        # Postselect to correct particle number AND alpha/beta split
        total_needed = n_samples * oversample_factor
        all_samples = self.model.sample(n_samples=total_needed, device=self.device)
        all_samples_np = all_samples.cpu().numpy()

        # Spin-orbital basis: even indices = alpha, odd indices = beta
        n_visible = all_samples_np.shape[1]
        alpha_indices = np.arange(0, n_visible, 2)
        beta_indices = np.arange(1, n_visible, 2)

        alpha_counts = all_samples_np[:, alpha_indices].sum(axis=1)
        beta_counts = all_samples_np[:, beta_indices].sum(axis=1)

        # Filter to correct alpha AND beta counts
        valid_mask = (alpha_counts == n_alpha) & (beta_counts == n_beta)
        valid_samples = all_samples_np[valid_mask]

        if len(valid_samples) >= n_samples:
            return valid_samples[:n_samples].astype("int8")
        else:
            return valid_samples.astype("int8") if len(valid_samples) > 0 else all_samples_np[:n_samples].astype("int8")


@dataclass
class ComplexFFNNSampler:
    """Adapter that uses a ComplexFFNNNQS model as a sampler.

    The complex NQS has separate amplitude and phase networks.
    Sampling is based on |ψ(x)|² (amplitude only), so it works
    the same as real NQS for sampling purposes.

    Includes optional postselection to enforce spin-sector conservation.
    For physical fermionic systems, we filter by (n_α, n_β) counts.

    In the Jordan-Wigner encoding with interleaved spin-orbitals:
    - Even indices (0, 2, 4, ...) are α spin-orbitals
    - Odd indices (1, 3, 5, ...) are β spin-orbitals
    """

    model: ComplexFFNNNQS
    device: torch.device
    n_alpha: int | None = None  # If set, postselect to this α electron count
    n_beta: int | None = None   # If set, postselect to this β electron count

    def sample(self, n_samples: int, oversample_factor: int = 50) -> np.ndarray:
        """Sample bitstrings from the model.

        Parameters
        ----------
        n_samples:
            Number of samples to return.
        oversample_factor:
            When postselecting, oversample by this factor to ensure enough valid samples.

        Returns
        -------
        samples:
            Array of shape (n_samples, n_visible) with 0/1 entries.
        """
        if self.n_alpha is None and self.n_beta is None:
            # No postselection, return raw samples
            samples = self.model.sample(n_samples=n_samples, device=self.device)
            return samples.cpu().numpy().astype("int8")

        # Postselect to correct (n_α, n_β) spin sector
        total_needed = n_samples * oversample_factor
        all_samples = self.model.sample(n_samples=total_needed, device=self.device)
        all_samples_np = all_samples.cpu().numpy()

        # Count α and β electrons (interleaved encoding)
        n_visible = all_samples_np.shape[1]
        alpha_indices = np.arange(0, n_visible, 2)
        beta_indices = np.arange(1, n_visible, 2)

        alpha_counts = all_samples_np[:, alpha_indices].sum(axis=1)
        beta_counts = all_samples_np[:, beta_indices].sum(axis=1)

        # Filter to correct (n_α, n_β) counts
        valid_mask = np.ones(len(all_samples_np), dtype=bool)
        if self.n_alpha is not None:
            valid_mask &= (alpha_counts == self.n_alpha)
        if self.n_beta is not None:
            valid_mask &= (beta_counts == self.n_beta)

        valid_samples = all_samples_np[valid_mask]

        if len(valid_samples) >= n_samples:
            return valid_samples[:n_samples].astype("int8")
        else:
            return valid_samples.astype("int8") if len(valid_samples) > 0 else all_samples_np[:n_samples].astype("int8")
