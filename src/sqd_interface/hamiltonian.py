"""Molecular Hamiltonian builders and bitstring mappings.

This module constructs molecular Hamiltonians using PySCF for electronic structure
calculations and performs Jordan-Wigner transformation to obtain qubit Hamiltonians
suitable for use with qiskit-addon-sqd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyscf import gto, scf
from qiskit.quantum_info import SparsePauliOp


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


@dataclass
class H2Hamiltonian:
    """Container for H2 Hamiltonian data.

    Attributes
    ----------
    hamiltonian:
        Qubit Hamiltonian as a SparsePauliOp (electronic part only).
    num_qubits:
        Number of qubits (spin orbitals).
    num_electrons:
        Number of electrons in the molecule.
    nuclear_repulsion:
        Nuclear repulsion energy (E_nuc).
    fci_energy:
        Full CI **total** energy = E_electronic + E_nuc.
    one_body_integrals:
        One-electron integrals in the spin-orbital basis.
    two_body_integrals:
        Two-electron integrals in the spin-orbital basis.

    Notes
    -----
    Energy reference convention:
    - The qubit Hamiltonian gives **electronic energy** when evaluated.
    - `fci_energy` is the **total** energy (electronic + nuclear repulsion).
    - Use `electronic_fci_energy` property for comparisons with VMC output.
    """

    hamiltonian: SparsePauliOp
    num_qubits: int
    num_electrons: int
    nuclear_repulsion: float
    fci_energy: float | None = None
    one_body_integrals: np.ndarray | None = None
    two_body_integrals: np.ndarray | None = None

    @property
    def electronic_fci_energy(self) -> float | None:
        """Electronic FCI energy (without nuclear repulsion).

        This is the correct reference for comparing VMC energies,
        since the qubit Hamiltonian gives electronic energy.
        """
        if self.fci_energy is None:
            return None
        return self.fci_energy - self.nuclear_repulsion

    @property
    def total_fci_energy(self) -> float | None:
        """Total FCI energy (electronic + nuclear repulsion).

        Alias for `fci_energy` for clarity.
        """
        return self.fci_energy


def _build_spin_orbital_integrals(
    h1e_mo: np.ndarray, eri_mo: np.ndarray, n_spatial: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MO integrals to spin-orbital basis.

    Parameters
    ----------
    h1e_mo:
        One-electron integrals in MO basis, shape (n_spatial, n_spatial).
    eri_mo:
        Two-electron integrals in MO basis (physicist's notation),
        shape (n_spatial, n_spatial, n_spatial, n_spatial).
    n_spatial:
        Number of spatial orbitals.

    Returns
    -------
    h1_spin:
        One-body integrals in spin-orbital basis, shape (2*n_spatial, 2*n_spatial).
    h2_spin:
        Two-body integrals in spin-orbital basis (antisymmetrized),
        shape (2*n_spatial, 2*n_spatial, 2*n_spatial, 2*n_spatial).
    """
    n_spin = 2 * n_spatial

    # One-body integrals in spin-orbital basis (interleaved: 0α, 0β, 1α, 1β, ...)
    h1_spin = np.zeros((n_spin, n_spin))
    for p in range(n_spatial):
        for q in range(n_spatial):
            # Alpha-alpha block
            h1_spin[2 * p, 2 * q] = h1e_mo[p, q]
            # Beta-beta block
            h1_spin[2 * p + 1, 2 * q + 1] = h1e_mo[p, q]

    # Two-body integrals in spin-orbital basis (antisymmetrized)
    h2_spin = np.zeros((n_spin, n_spin, n_spin, n_spin))
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    # Alpha-alpha-alpha-alpha
                    h2_spin[2 * p, 2 * q, 2 * r, 2 * s] = eri_mo[p, q, r, s] - eri_mo[p, q, s, r]
                    # Beta-beta-beta-beta
                    h2_spin[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = (
                        eri_mo[p, q, r, s] - eri_mo[p, q, s, r]
                    )
                    # Alpha-alpha-beta-beta
                    h2_spin[2 * p, 2 * q, 2 * r + 1, 2 * s + 1] = eri_mo[p, q, r, s]
                    # Beta-beta-alpha-alpha
                    h2_spin[2 * p + 1, 2 * q + 1, 2 * r, 2 * s] = eri_mo[p, q, r, s]
                    # Alpha-beta-alpha-beta
                    h2_spin[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = eri_mo[p, q, r, s]
                    # Beta-alpha-beta-alpha
                    h2_spin[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = eri_mo[p, q, r, s]
                    # Alpha-beta-beta-alpha
                    h2_spin[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = -eri_mo[p, q, s, r]
                    # Beta-alpha-alpha-beta
                    h2_spin[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = -eri_mo[p, q, s, r]

    return h1_spin, h2_spin


def build_h2_hamiltonian_12bit(cfg: H2Config) -> H2Hamiltonian:
    """Construct H2 Hamiltonian using qiskit-nature's ElectronicStructureProblem.

    This function:
    1. Builds H2 molecule using qiskit-nature drivers
    2. Creates ElectronicStructureProblem with proper second quantized operators
    3. Applies Jordan-Wigner transformation
    4. Optionally pads to target bit_depth qubits
    5. Computes FCI reference energy for validation

    Parameters
    ----------
    cfg:
        H2 configuration specifying bond length and target bit depth.

    Returns
    -------
    H2Hamiltonian:
        Container with qubit Hamiltonian and molecular data.

    Notes
    -----
    H2 in STO-3G has 2 spatial orbitals -> 4 spin orbitals -> 4 qubits minimum.
    For bit_depth > 4, the Hamiltonian is padded with identity operators.
    """
    # Try using qiskit-nature for full workflow
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.units import DistanceUnit

        # Build H2 molecule using qiskit-nature
        driver = PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {cfg.bond_length}",
            basis="sto-3g",
            unit=DistanceUnit.ANGSTROM,
        )

        # Get electronic structure problem
        problem = driver.run()

        # Get the fermionic Hamiltonian
        hamiltonian_ferm = problem.hamiltonian

        # Get reference data
        nuclear_repulsion = hamiltonian_ferm.nuclear_repulsion_energy
        num_particles = problem.num_particles
        num_spatial_orbitals = problem.num_spatial_orbitals
        n_spin = 2 * num_spatial_orbitals

        # Map to qubits using Jordan-Wigner
        mapper = JordanWignerMapper()
        qubit_op = mapper.map(hamiltonian_ferm.second_q_op())

        # Pad to target bit depth if needed
        if cfg.bit_depth > n_spin:
            # Extend Pauli strings with identities
            padded_pauli_strings = []
            for pauli_str in qubit_op.paulis.to_labels():
                padded_pauli_strings.append(pauli_str + "I" * (cfg.bit_depth - n_spin))
            qubit_op = SparsePauliOp(padded_pauli_strings, qubit_op.coeffs)

        # Compute FCI energy using PySCF directly
        from pyscf import gto, scf, fci as pyscf_fci

        mol = gto.M(
            atom=f"H 0 0 0; H 0 0 {cfg.bond_length}",
            basis="sto-3g",
            unit="angstrom",
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.kernel()
        fci_solver = pyscf_fci.FCI(mol, mf.mo_coeff)
        fci_energy = fci_solver.kernel()[0]

        # Get integrals for SQD
        # We need to extract one- and two-body integrals in spin-orbital basis
        # For now, fall back to manual extraction
        mo_coeff = mf.mo_coeff
        h1e_mo = mo_coeff.T @ mf.get_hcore() @ mo_coeff
        eri_mo = mol.intor("int2e")
        eri_mo = np.einsum("pi,qj,rk,sl,ijkl->pqrs", mo_coeff, mo_coeff, mo_coeff, mo_coeff, eri_mo)

        n_spatial = mol.nao_nr()
        h1_spin, h2_spin = _build_spin_orbital_integrals(h1e_mo, eri_mo, n_spatial)

        return H2Hamiltonian(
            hamiltonian=qubit_op,
            num_qubits=cfg.bit_depth if cfg.bit_depth > n_spin else n_spin,
            num_electrons=sum(num_particles),
            nuclear_repulsion=nuclear_repulsion,
            fci_energy=fci_energy,
            one_body_integrals=h1_spin,
            two_body_integrals=h2_spin,
        )

    except (ImportError, Exception) as e:
        print(f"WARNING: qiskit-nature failed ({type(e).__name__}: {e}), using manual implementation")
        # Fall back to manual implementation below
        pass

    # Manual implementation (fallback)
    # Build H2 molecule
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {cfg.bond_length}",
        basis="sto-3g",
        unit="angstrom",
        verbose=0,
    )

    # Run RHF calculation for integrals
    mf = scf.RHF(mol)
    mf.kernel()

    # Get integrals in MO basis
    mo_coeff = mf.mo_coeff
    h1e_mo = mo_coeff.T @ mf.get_hcore() @ mo_coeff  # One-electron integrals
    eri_mo = mol.intor("int2e")  # Two-electron integrals in AO basis (physicist's notation)

    # Transform ERIs to MO basis: eri_mo[p,q,r,s] = (pq|rs) in physicist's notation
    eri_mo = np.einsum("pi,qj,rk,sl,ijkl->pqrs", mo_coeff, mo_coeff, mo_coeff, mo_coeff, eri_mo)

    # Convert to spin-orbital basis
    n_spatial = mol.nao_nr()  # Number of spatial orbitals (2 for H2 in STO-3G)
    n_spin = 2 * n_spatial  # Number of spin orbitals (4 for H2)
    h1_spin, h2_spin = _build_spin_orbital_integrals(h1e_mo, eri_mo, n_spatial)

    # Jordan-Wigner transformation to qubit operators
    # Use Qiskit's FermionicOp to ensure correct transformation
    try:
        from qiskit_nature.second_q.operators import FermionicOp
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        def jordan_wigner_transform_nature() -> SparsePauliOp:
            """Apply Jordan-Wigner using qiskit-nature for correctness."""
            # Build fermionic Hamiltonian using qiskit-nature
            fermionic_op_dict = {}

            # Constant term (nuclear repulsion)
            fermionic_op_dict[""] = mol.energy_nuc()

            # One-body terms: h1[p,q] a†_p a_q -> "+_p -_q"
            for p in range(n_spin):
                for q in range(n_spin):
                    if abs(h1_spin[p, q]) < 1e-12:
                        continue
                    key = f"+_{p} -_{q}"
                    fermionic_op_dict[key] = fermionic_op_dict.get(key, 0.0) + h1_spin[p, q]

            # Two-body terms: 0.5 * h2[p,q,r,s] a†_p a†_q a_s a_r -> "+_p +_q -_s -_r"
            # Note: h2_spin is antisymmetrized
            for p in range(n_spin):
                for q in range(n_spin):
                    for r in range(n_spin):
                        for s in range(n_spin):
                            if abs(h2_spin[p, q, r, s]) < 1e-12:
                                continue
                            # Factor of 0.5 for double counting in fermionic Hamiltonian
                            key = f"+_{p} +_{q} -_{s} -_{r}"
                            fermionic_op_dict[key] = fermionic_op_dict.get(key, 0.0) + 0.5 * h2_spin[p, q, r, s]

            # Create FermionicOp
            fermionic_op = FermionicOp(fermionic_op_dict, num_spin_orbitals=n_spin)

            # Apply Jordan-Wigner mapping
            mapper = JordanWignerMapper()
            qubit_op = mapper.map(fermionic_op)

            return qubit_op

        jordan_wigner_transform = jordan_wigner_transform_nature

    except ImportError:
        # Fallback: manual Jordan-Wigner (simplified, may not be complete)
        from qiskit.quantum_info import Pauli

        def jordan_wigner_transform_manual() -> SparsePauliOp:
            """Manual Jordan-Wigner (simplified - only handles number operators correctly)."""
            pauli_list = []
            coeff_list = []

            # Constant term (nuclear repulsion)
            nuclear_repulsion = mol.energy_nuc()
            pauli_list.append(Pauli("I" * n_spin))
            coeff_list.append(nuclear_repulsion)

            # One-body terms: h1[p,q] a†_p a_q
            for p in range(n_spin):
                for q in range(n_spin):
                    if abs(h1_spin[p, q]) < 1e-12:
                        continue

                    if p == q:
                        # Number operator: a†_p a_p = (I - Z_p) / 2
                        pauli_z = ["I"] * n_spin
                        pauli_z[p] = "Z"
                        pauli_list.append(Pauli("".join(pauli_z)))
                        coeff_list.append(-0.5 * h1_spin[p, q])

                        pauli_list.append(Pauli("I" * n_spin))
                        coeff_list.append(0.5 * h1_spin[p, q])
                    else:
                        # Excitation/de-excitation: a†_p a_q
                        min_idx, max_idx = min(p, q), max(p, q)

                        # XX term
                        pauli_xx = ["I"] * n_spin
                        pauli_xx[p] = "X"
                        pauli_xx[q] = "X"
                        for k in range(min_idx + 1, max_idx):
                            pauli_xx[k] = "Z"
                        pauli_list.append(Pauli("".join(pauli_xx)))
                        coeff_list.append(0.5 * h1_spin[p, q])

                        # YY term
                        pauli_yy = ["I"] * n_spin
                        pauli_yy[p] = "Y"
                        pauli_yy[q] = "Y"
                        for k in range(min_idx + 1, max_idx):
                            pauli_yy[k] = "Z"
                        pauli_list.append(Pauli("".join(pauli_yy)))
                        coeff_list.append(0.5 * h1_spin[p, q])

            # Two-body terms: only number-number terms (incomplete!)
            for p in range(n_spin):
                for q in range(n_spin):
                    for r in range(n_spin):
                        for s in range(n_spin):
                            coeff = 0.125 * h2_spin[p, q, r, s]
                            if abs(coeff) < 1e-12:
                                continue

                            # Number-number terms: when p=r and q=s
                            if p == r and q == s:
                                if p == q:
                                    # n_p^2 = n_p (Pauli exclusion)
                                    pauli_z = ["I"] * n_spin
                                    pauli_z[p] = "Z"
                                    pauli_list.append(Pauli("".join(pauli_z)))
                                    coeff_list.append(-0.25 * h2_spin[p, q, r, s])

                                    pauli_list.append(Pauli("I" * n_spin))
                                    coeff_list.append(0.25 * h2_spin[p, q, r, s])
                                else:
                                    # n_p n_q
                                    pauli_list.append(Pauli("I" * n_spin))
                                    coeff_list.append(0.25 * h2_spin[p, q, r, s])

                                    pauli_z_p = ["I"] * n_spin
                                    pauli_z_p[p] = "Z"
                                    pauli_list.append(Pauli("".join(pauli_z_p)))
                                    coeff_list.append(-0.25 * h2_spin[p, q, r, s])

                                    pauli_z_q = ["I"] * n_spin
                                    pauli_z_q[q] = "Z"
                                    pauli_list.append(Pauli("".join(pauli_z_q)))
                                    coeff_list.append(-0.25 * h2_spin[p, q, r, s])

                                    pauli_zz = ["I"] * n_spin
                                    pauli_zz[p] = "Z"
                                    pauli_zz[q] = "Z"
                                    pauli_list.append(Pauli("".join(pauli_zz)))
                                    coeff_list.append(0.25 * h2_spin[p, q, r, s])

            print("WARNING: Using manual Jordan-Wigner transformation which may be incomplete!")
            print("         Install qiskit-nature for full correctness.")
            return SparsePauliOp(pauli_list, np.array(coeff_list))

        jordan_wigner_transform = jordan_wigner_transform_manual

    # Build Hamiltonian
    nuclear_repulsion = mol.energy_nuc()
    hamiltonian = jordan_wigner_transform()
    hamiltonian = hamiltonian.simplify(atol=1e-12)  # Combine like terms

    # Pad to target bit depth if needed
    if cfg.bit_depth > n_spin:
        # Extend Pauli strings with identities
        padded_pauli_strings = []
        for pauli_str in hamiltonian.paulis.to_labels():
            padded_pauli_strings.append(pauli_str + "I" * (cfg.bit_depth - n_spin))
        hamiltonian = SparsePauliOp(padded_pauli_strings, hamiltonian.coeffs)

    # Compute FCI energy for reference
    from pyscf import fci as pyscf_fci

    fci_solver = pyscf_fci.FCI(mol, mo_coeff)
    fci_energy = fci_solver.kernel()[0]

    return H2Hamiltonian(
        hamiltonian=hamiltonian,
        num_qubits=cfg.bit_depth if cfg.bit_depth > n_spin else n_spin,
        num_electrons=mol.nelectron,
        nuclear_repulsion=nuclear_repulsion,
        fci_energy=fci_energy,
        one_body_integrals=h1_spin,
        two_body_integrals=h2_spin,
    )
