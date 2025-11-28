"""Transformer-based Neural Quantum State (NQS) models.

This file defines a *skeleton* for an autoregressive Transformer NQS that models
a distribution over bitstrings σ ∈ {0,1}^N. The goal is to approximate the
(marginal) distribution of important configurations so that SQD can use these
samples for subspace construction.

You can extend this with more advanced architectures later (e.g. multi-head
self-attention, positional encodings tailored to orbitals, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from einops import rearrange


@dataclass
class TransformerNQSConfig:
    """Configuration for a Transformer-based NQS.

    Attributes
    ----------
    n_visible:
        Number of visible units (e.g. spin orbitals / qubits).
    d_model:
        Model dimension for the Transformer.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of Transformer encoder layers.
    dropout:
        Dropout probability inside the Transformer.
    """

    n_visible: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


class TransformerNQS(nn.Module):
    """Autoregressive Transformer NQS over bitstrings.

    This model takes as input a batch of bitstrings (0/1) and predicts logits
    for each position. For now we treat the sequence as length `n_visible` with
    a trivial positional encoding. You can later replace this with orbital-aware
    encodings or additional symmetry constraints.
    """

    def __init__(self, config: TransformerNQSConfig) -> None:
        super().__init__()
        self.config = config
        self.n_visible = config.n_visible

        self.input_embed = nn.Linear(1, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.output_head = nn.Linear(config.d_model, 1)

        # simple learnable positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_visible, config.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-site logits for p(σ_i | σ_{<i}).

        Parameters
        ----------
        x:
            Tensor of shape (batch, n_visible) containing 0/1 values.

        Returns
        -------
        logits:
            Tensor of shape (batch, n_visible) with real-valued logits.
        """
        b, n = x.shape
        assert n == self.n_visible, "Unexpected sequence length"

        # (batch, n_visible, 1) -> (batch, n_visible, d_model)
        h = self.input_embed(x.unsqueeze(-1))
        h = h + self.pos_emb[:, :n, :]

        # causal masking can be added later; for now we use full self-attention
        h = self.encoder(h)
        logits = self.output_head(h).squeeze(-1)
        return logits

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) under the model (teacher-forced).

        This is a simple teacher-forced cross-entropy style computation. For
        a fully autoregressive model you may want to add an explicit causal
        mask and compute p(σ_i | σ_{<i}).
        """
        logits = self.forward(x)
        log_p = -nn.functional.binary_cross_entropy_with_logits(
            logits, x.float(), reduction="none"
        )
        # sum over sites
        return log_p.sum(dim=-1)

    @torch.no_grad()
    def sample(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Autoregressively sample bitstrings from the model (very simple version)."""
        device = device or next(self.parameters()).device
        x = torch.zeros(n_samples, self.n_visible, device=device)
        for i in range(self.n_visible):
            logits = self.forward(x)
            probs = torch.sigmoid(logits[:, i])
            x[:, i] = torch.bernoulli(probs)
        return x
