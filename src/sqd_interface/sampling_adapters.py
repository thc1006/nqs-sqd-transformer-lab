"""Adapters that connect samplers (NQS or baselines) to the SQD interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from src.nqs_models.ffn_nqs import FFNNNQS
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

    Currently uses the naive `.sample` method implemented in `FFNNNQS`. You
    should replace this with a proper MCMC or energy-guided sampler once the
    model is trained.
    """

    model: FFNNNQS
    device: torch.device

    def sample(self, n_samples: int) -> np.ndarray:
        samples = self.model.sample(n_samples=n_samples, device=self.device)
        return samples.cpu().numpy().astype("int8")


@dataclass
class TransformerSampler:
    """Adapter that uses a TransformerNQS model as an autoregressive sampler."""

    model: TransformerNQS
    device: torch.device

    def sample(self, n_samples: int) -> np.ndarray:
        samples = self.model.sample(n_samples=n_samples, device=self.device)
        return samples.cpu().numpy().astype("int8")
