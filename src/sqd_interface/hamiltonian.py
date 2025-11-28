"""Molecular Hamiltonian builders and bitstring mappings.

Currently contains a placeholder for an H2 Hamiltonian in a small basis.
You can extend it using PySCF, qiskit-nature, or hard-coded integrals for
toy models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class H2Config:
    """Configuration for a simple H2 Hamiltonian.

    Attributes
    ----------
    bond_length:
        H–H bond length in Angstrom.
    bit_depth:
        Target bit depth / number of qubits used for encoding.
    """

    bond_length: float = 0.74
    bit_depth: int = 12


def build_h2_hamiltonian_12bit(cfg: H2Config) -> Any:
    """Placeholder for constructing an H2 Hamiltonian and mapping.

    Returns an opaque object for now (e.g. later a qiskit `SparsePauliOp`
    plus metadata). For initial development you can just return a simple toy
    Hamiltonian (e.g. 2-qubit ZZ + XX model) and upgrade later.
    """
    raise NotImplementedError("build_h2_hamiltonian_12bit is not implemented yet.")
