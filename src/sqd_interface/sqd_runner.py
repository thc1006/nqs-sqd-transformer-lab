"""Thin wrapper around qiskit-addon-sqd.

Expose a simple function that takes:

- a Hamiltonian object / integrals,
- a list or array of bitstring samples,

and returns estimated ground-state energies and diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def run_sqd_on_samples(
    hamiltonian: Any,
    samples: Sequence[int],
    max_subspace_dim: int = 256,
) -> Dict[str, Any]:
    """Run SQD given a Hamiltonian and a set of samples.

    This is currently just a stub that returns dummy values so that the
    experiment scripts can run end-to-end without errors. You should replace
    the body with actual calls into `qiskit-addon-sqd`.

    Parameters
    ----------
    hamiltonian:
        Placeholder Hamiltonian (to be defined).
    samples:
        Iterable of integer bitstring encodings or raw bitstrings.
    max_subspace_dim:
        Maximum dimension for the SQD subspace.

    Returns
    -------
    result:
        A dictionary with at least:
        - 'energy_estimate': float
        - 'subspace_dim': int
    """
    # Dummy behavior: pretend we estimated -5.6 Ha with a small subspace.
    return {
        "energy_estimate": -5.6,
        "subspace_dim": min(max_subspace_dim, len(list(samples))),
    }
