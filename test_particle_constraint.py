#!/usr/bin/env python
"""Test particle number constraint for VMC training.

This script tests whether restricting the configuration space to fixed particle
number (N=2 for H₂) can break through the ~20.5 mHa accuracy floor.

Hypothesis: The 20.5 mHa floor is caused by NQS assigning non-zero amplitudes
to unphysical configurations (e.g., configurations with 0, 1, 3, or 4 electrons
instead of the correct 2 electrons).

Solution: Only use configurations with exactly N=2 particles (electrons).
For 4-qubit H₂, this reduces configuration space from 2^4=16 to C(4,2)=6.

Usage:
    python test_particle_constraint.py
"""

import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Enable TF32 for faster matmul on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TestResult:
    """Container for test results."""
    name: str
    n_configs: int
    electronic_fci: float
    final_energy: float
    error_mha: float
    training_time: float
    chemical_accuracy: bool


def run_particle_constraint_test(
    n_qubits: int = 4,
    n_electrons: int = 2,
    n_epochs: int = 300,
    hidden_size: int = 64,
    n_layers: int = 2,
    learning_rate: float = 5e-3,
    use_particle_constraint: bool = True,
    verbose: bool = True,
) -> TestResult:
    """Run VMC test with or without particle number constraint.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    n_electrons:
        Number of electrons (particles).
    n_epochs:
        Number of training epochs.
    hidden_size:
        FFNN hidden layer size.
    n_layers:
        FFNN number of layers.
    learning_rate:
        Optimizer learning rate.
    use_particle_constraint:
        If True, only use configurations with exactly n_electrons particles.
    verbose:
        Print detailed output.

    Returns
    -------
    TestResult with all metrics.
    """
    from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
    from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    from src.training.vmc_trainer import VMCTrainer, VMCConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    test_name = f"{'Constrained' if use_particle_constraint else 'Unconstrained'} N={n_electrons}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"VMC Test: {test_name}")
        print(f"{'='*60}")

    # Build Hamiltonian
    h2_config = H2Config(bond_length=0.74, bit_depth=n_qubits)
    hamiltonian = build_h2_hamiltonian_12bit(h2_config)
    electronic_fci = hamiltonian.electronic_fci_energy

    if verbose:
        print(f"\nEnergy Reference:")
        print(f"  Electronic FCI: {electronic_fci:.6f} Ha")
        print(f"  Chemical accuracy: 1.6 mHa = 0.0016 Ha")

    # Create model
    model_config = FFNNNQSConfig(
        n_visible=n_qubits,
        n_hidden=hidden_size,
        n_layers=n_layers,
    )
    model = FFNNNQS(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\nModel: FFNNNQS({hidden_size}, {n_layers} layers)")
        print(f"Parameters: {n_params:,}")
        print(f"Device: {device}")

    # Configure VMC with or without particle constraint
    vmc_config = VMCConfig(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        use_exact=True,
        scheduler_type="cosine",
        warmup_epochs=max(10, n_epochs // 10),
        clip_grad_norm=1.0,
        # Particle number constraint settings
        use_particle_constraint=use_particle_constraint,
        n_particles=n_electrons,
    )

    # Train
    start_time = time.time()
    trainer = VMCTrainer(model, hamiltonian, vmc_config)

    # Get config count for reporting
    n_configs = len(trainer.all_configs)

    if verbose:
        print(f"\nConfiguration space:")
        print(f"  Total configs (2^{n_qubits}): {2**n_qubits}")
        print(f"  Used configs: {n_configs}")
        if use_particle_constraint:
            print(f"  Constraint: exactly {n_electrons} particles")

    history = trainer.train()
    training_time = time.time() - start_time

    # Get final energy
    final_energy = history["energy"][-1]
    error = abs(final_energy - electronic_fci)
    error_mha = error * 1000
    chemical_accuracy = error < 0.0016

    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS: {test_name}")
        print(f"{'='*60}")
        print(f"Final electronic energy:  {final_energy:.6f} Ha")
        print(f"Electronic FCI:           {electronic_fci:.6f} Ha")
        print(f"Error:                    {error:.6f} Ha ({error_mha:.3f} mHa)")
        print(f"Training time:            {training_time:.1f} s")
        print(f"Chemical accuracy (1.6 mHa): {'ACHIEVED' if chemical_accuracy else 'NOT achieved'}")
        print(f"{'='*60}")

    return TestResult(
        name=test_name,
        n_configs=n_configs,
        electronic_fci=electronic_fci,
        final_energy=final_energy,
        error_mha=error_mha,
        training_time=training_time,
        chemical_accuracy=chemical_accuracy,
    )


def main():
    """Run comparison experiment."""
    print("=" * 70)
    print("VMC Particle Number Constraint Test")
    print("=" * 70)
    print("\nHypothesis: Restricting to N=2 configurations reduces error")
    print("from ~20.5 mHa to potentially < 1.6 mHa (chemical accuracy)")

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nWARNING: No GPU available, using CPU")

    results = []

    # Test 1: Without particle constraint (baseline)
    print("\n" + "=" * 70)
    print("TEST 1: Without Particle Constraint (Baseline)")
    print("=" * 70)
    result_unconstrained = run_particle_constraint_test(
        n_qubits=4,
        n_electrons=2,
        n_epochs=300,
        hidden_size=64,
        n_layers=2,
        learning_rate=5e-3,
        use_particle_constraint=False,
        verbose=True,
    )
    results.append(result_unconstrained)

    # Test 2: With particle constraint N=2
    print("\n" + "=" * 70)
    print("TEST 2: With Particle Constraint N=2")
    print("=" * 70)
    result_constrained = run_particle_constraint_test(
        n_qubits=4,
        n_electrons=2,
        n_epochs=300,
        hidden_size=64,
        n_layers=2,
        learning_rate=5e-3,
        use_particle_constraint=True,
        verbose=True,
    )
    results.append(result_constrained)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Particle Number Constraint Effect")
    print("=" * 70)
    print(f"{'Test':<25} {'Configs':<10} {'Error (mHa)':<15} {'Chem Acc':<12}")
    print("-" * 70)
    for r in results:
        status = "YES" if r.chemical_accuracy else "NO"
        print(f"{r.name:<25} {r.n_configs:<10} {r.error_mha:<15.3f} {status:<12}")
    print("=" * 70)

    # Analysis
    improvement = result_unconstrained.error_mha - result_constrained.error_mha
    improvement_pct = (improvement / result_unconstrained.error_mha) * 100

    print(f"\nImprovement: {improvement:.3f} mHa ({improvement_pct:.1f}%)")

    if result_constrained.chemical_accuracy:
        print("\n*** SUCCESS: Chemical accuracy achieved with particle constraint! ***")
    elif result_constrained.error_mha < result_unconstrained.error_mha:
        print(f"\n** Partial success: Error reduced but still > 1.6 mHa **")
        print("   Possible next steps:")
        print("   1. Implement complex wavefunction (sign structure)")
        print("   2. Use spin-adapted configurations")
        print("   3. Use neural network with antisymmetry built-in")
    else:
        print("\nNo improvement from particle constraint alone.")
        print("The sign problem may be the primary bottleneck.")

    return results


if __name__ == "__main__":
    main()
