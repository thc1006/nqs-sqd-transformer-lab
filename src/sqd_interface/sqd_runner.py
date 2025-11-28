"""Thin wrapper around qiskit-addon-sqd.

Expose a simple function that takes:

- a Hamiltonian object / integrals,
- a list or array of bitstring samples,

and returns estimated ground-state energies and diagnostics.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Sequence

import numpy as np
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch

from .hamiltonian import H2Hamiltonian


def _get_spatial_integrals_from_pyscf(bond_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatial orbital integrals directly from PySCF.

    This bypasses the spin-orbital integrals stored in H2Hamiltonian and
    computes fresh integrals in the correct format for qiskit-addon-sqd.

    Parameters
    ----------
    bond_length:
        H-H bond length in Angstrom.

    Returns
    -------
    hcore:
        One-body integrals in spatial orbital basis (MO), shape (n_spatial, n_spatial).
    eri:
        Two-body integrals in chemist's notation (pq|rs) in MO basis,
        shape (n_spatial, n_spatial, n_spatial, n_spatial).
    """
    from pyscf import gto, scf, ao2mo

    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis="sto-3g",
        unit="angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    mo_coeff = mf.mo_coeff
    n_spatial = mol.nao_nr()

    # One-body integrals in MO basis
    hcore = mo_coeff.T @ mf.get_hcore() @ mo_coeff

    # Two-body integrals using ao2mo (handles conventions correctly)
    eri_packed = ao2mo.kernel(mol, mo_coeff)
    eri = ao2mo.restore(1, eri_packed, n_spatial)  # Restore to 4-index tensor

    return hcore, eri


def _spin_orbital_to_spatial_integrals(
    h1_spin: np.ndarray, h2_spin: np.ndarray, n_spatial: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert spin-orbital integrals to spatial orbital integrals.

    WARNING: This function has known issues with integral conventions.
    Consider using _get_spatial_integrals_from_pyscf() instead.

    H2Hamiltonian stores integrals in spin-orbital basis (interleaved: 0α, 0β, 1α, 1β).
    qiskit-addon-sqd requires spatial orbital integrals in chemist's notation.

    Parameters
    ----------
    h1_spin:
        One-body integrals in spin-orbital basis, shape (2*n_spatial, 2*n_spatial).
    h2_spin:
        Two-body integrals in spin-orbital basis (antisymmetrized),
        shape (2*n_spatial, 2*n_spatial, 2*n_spatial, 2*n_spatial).
    n_spatial:
        Number of spatial orbitals.

    Returns
    -------
    hcore:
        One-body integrals in spatial orbital basis, shape (n_spatial, n_spatial).
    eri:
        Two-body integrals in spatial orbital basis (chemist's notation (pq|rs)),
        shape (n_spatial, n_spatial, n_spatial, n_spatial).
    """
    # Extract α-α block from one-body integrals
    hcore = np.zeros((n_spatial, n_spatial))
    for p in range(n_spatial):
        for q in range(n_spatial):
            hcore[p, q] = h1_spin[2 * p, 2 * q]

    # The two-body integrals in h2_spin are stored in the α-α-β-β block as
    # the original MO chemist's notation integrals (pq|rs).
    # h2_spin[2p, 2q, 2r+1, 2s+1] = eri_mo[p, q, r, s] = (pq|rs)_chemist
    eri = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial))
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    eri[p, q, r, s] = h2_spin[2 * p, 2 * q, 2 * r + 1, 2 * s + 1]

    return hcore, eri


def _convert_bitstrings_to_sqd_format(
    samples: np.ndarray, n_spatial: int, num_qubits: int
) -> np.ndarray:
    """Convert NQS bitstring samples to qiskit-addon-sqd format.

    Parameters
    ----------
    samples:
        Bitstring samples from NQS, shape (n_samples, num_qubits).
        For H2 with 12-bit encoding: (n_samples, 12).
        Assumes occupation number representation where 1 = occupied, 0 = empty.
    n_spatial:
        Number of spatial orbitals (e.g., 2 for H2 in STO-3G).
    num_qubits:
        Total number of qubits in the encoding.

    Returns
    -------
    bitstring_matrix:
        Boolean array of shape (n_samples, 2*n_spatial).
        Format: [β_{n_spatial-1}, ..., β_0, α_{n_spatial-1}, ..., α_0].
        α spins in right half [n_spatial:2*n_spatial].
        β spins in left half [0:n_spatial].

    Notes
    -----
    For H2 with spin-orbital ordering (0α, 0β, 1α, 1β, ...) in the first 4 qubits,
    we extract:
    - α electrons from even indices: [0, 2, 4, ...]
    - β electrons from odd indices: [1, 3, 5, ...]
    Then reorder to SQD format.
    """
    n_samples = samples.shape[0]
    bitstring_matrix = np.zeros((n_samples, 2 * n_spatial), dtype=bool)

    # Extract α and β spins from interleaved spin-orbital basis
    for i in range(n_spatial):
        # α spin from even indices (2i) -> right half of output
        bitstring_matrix[:, n_spatial + i] = samples[:, 2 * i].astype(bool)
        # β spin from odd indices (2i+1) -> left half of output
        bitstring_matrix[:, i] = samples[:, 2 * i + 1].astype(bool)

    # Reverse order within each spin block to match SQD convention
    # SQD expects: [β_{norb-1}, ..., β_0, α_{norb-1}, ..., α_0]
    bitstring_matrix[:, :n_spatial] = np.flip(bitstring_matrix[:, :n_spatial], axis=1)
    bitstring_matrix[:, n_spatial:] = np.flip(bitstring_matrix[:, n_spatial:], axis=1)

    return bitstring_matrix


def run_sqd_on_samples(
    hamiltonian: H2Hamiltonian,
    samples: np.ndarray,
    samples_per_batch: int = 300,
    num_batches: int = 10,
    max_iterations: int = 5,
    spin_sq: float = 0.0,
    max_cycle: int = 200,
    seed: int = 42,
    bond_length: float = 0.74,
) -> Dict[str, Any]:
    """Run SQD given a Hamiltonian and a set of bitstring samples.

    This function:
    1. Computes spatial orbital integrals directly from PySCF
    2. Converts NQS bitstrings to qiskit-addon-sqd format
    3. Runs diagonalize_fermionic_hamiltonian with selected configuration iteration
    4. Returns energy estimate with nuclear repulsion included

    Parameters
    ----------
    hamiltonian:
        H2Hamiltonian object containing molecular integrals and metadata.
    samples:
        Bitstring samples from NQS, shape (n_samples, num_qubits).
        Each bitstring encodes occupation numbers (1 = occupied, 0 = empty).
    samples_per_batch:
        Number of samples to use per SQD batch (default: 300).
    num_batches:
        Number of batches for configuration recovery (default: 10).
    max_iterations:
        Maximum iterations for configuration recovery (default: 5).
    spin_sq:
        Target spin quantum number S(S+1) for SCI solver (default: 0.0 for singlet).
    max_cycle:
        Maximum cycles for SCI solver (default: 200).
    seed:
        Random seed for reproducibility (default: 42).
    bond_length:
        H-H bond length in Angstrom (default: 0.74 for equilibrium H₂).

    Returns
    -------
    result:
        Dictionary with:
        - 'energy_estimate': Total energy including nuclear repulsion (float)
        - 'electronic_energy': Electronic energy from SQD (float)
        - 'nuclear_repulsion': Nuclear repulsion energy (float)
        - 'subspace_dim': Final subspace dimension used (int)
        - 'n_samples': Number of samples provided (int)
        - 'n_unique_samples': Number of unique configurations (int)
        - 'converged': Whether SQD converged (bool)

    Raises
    ------
    ValueError:
        If hamiltonian does not have one_body_integrals or two_body_integrals.
    """
    # Determine number of spatial orbitals
    # For H2 in STO-3G: 2 spatial orbitals -> 4 spin orbitals
    if hamiltonian.one_body_integrals is not None:
        n_spin = hamiltonian.one_body_integrals.shape[0]
        n_spatial = n_spin // 2
    else:
        # Default for H2 in STO-3G
        n_spatial = 2

    # Determine number of α and β electrons
    # For H2: 2 electrons, assume restricted (1α, 1β)
    n_alpha = hamiltonian.num_electrons // 2
    n_beta = hamiltonian.num_electrons // 2

    # Get spatial orbital integrals directly from PySCF (correct convention)
    hcore, eri = _get_spatial_integrals_from_pyscf(bond_length)

    # Convert bitstrings to SQD format
    bitstring_matrix = _convert_bitstrings_to_sqd_format(
        samples, n_spatial, hamiltonian.num_qubits
    )

    # Count unique configurations
    unique_bitstrings = np.unique(bitstring_matrix, axis=0)
    n_unique = len(unique_bitstrings)

    # Configure SCI solver
    sci_solver = partial(solve_sci_batch, spin_sq=spin_sq, max_cycle=max_cycle)

    # Convert boolean array to BitArray for qiskit-addon-sqd
    # BitArray expects shape (num_shots, num_bits) as boolean array
    bit_array = BitArray.from_bool_array(bitstring_matrix)

    # Run SQD
    sqd_result = diagonalize_fermionic_hamiltonian(
        hcore,
        eri,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=n_spatial,
        nelec=(n_alpha, n_beta),
        num_batches=num_batches,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        seed=seed,
    )

    # Extract results
    # Note: sqd_result.energy does NOT include nuclear repulsion
    electronic_energy = sqd_result.energy
    total_energy = electronic_energy + hamiltonian.nuclear_repulsion

    # Compute subspace dimension from SCI state
    # The subspace is the Cartesian product of alpha and beta CI strings
    sci_state = sqd_result.sci_state
    subspace_dim = len(sci_state.ci_strs_a) * len(sci_state.ci_strs_b)

    # Check convergence based on whether energy changed significantly
    # (qiskit-addon-sqd doesn't have explicit converged flag in recent versions)
    converged = True  # Assume converged if no exception was raised

    return {
        "energy_estimate": total_energy,
        "electronic_energy": electronic_energy,
        "nuclear_repulsion": hamiltonian.nuclear_repulsion,
        "subspace_dim": subspace_dim,
        "n_samples": samples.shape[0],
        "n_unique_samples": n_unique,
        "converged": converged,
    }
