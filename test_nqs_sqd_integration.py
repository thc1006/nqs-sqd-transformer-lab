#!/usr/bin/env python
"""Test NQS-SQD integration: compare random vs trained NQS sampling.

This script tests the main hypothesis of the project:
Can a trained NQS sampler achieve better sample efficiency than random sampling
for Sample-based Quantum Diagonalization (SQD)?

Experiment design:
1. Train a ComplexFFNNNQS model on 4-qubit H₂ to achieve chemical accuracy
2. Generate samples from:
   (a) Random sampler (baseline)
   (b) Trained NQS (ComplexFFNNNQS)
3. Run SQD with each sampler at various sample counts
4. Compare sample efficiency to reach target accuracy

Expected result: NQS should require fewer samples to reach the same accuracy
because it samples from the learned |ψ|² distribution concentrated on
important configurations.

Usage:
    python test_nqs_sqd_integration.py
"""

import time
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Suppress output during training
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Enable TF32 for faster matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class SQDResult:
    """Container for SQD experiment results."""
    sampler_name: str
    n_samples: int
    energy_estimate: float
    electronic_energy: float
    fci_energy: float
    error_mha: float
    n_unique_samples: int
    subspace_dim: int
    time_seconds: float


def train_complex_nqs(
    n_qubits: int = 4,
    n_epochs: int = 1000,
    hidden_size: int = 32,
    n_layers: int = 2,
    learning_rate: float = 5e-3,
    n_particles: int = 2,
    verbose: bool = True,
) -> Tuple["ComplexFFNNNQS", float]:
    """Train a ComplexFFNNNQS model on H₂.

    Returns the trained model and its VMC energy error in mHa.
    """
    from src.nqs_models.ffn_nqs import ComplexFFNNNQS, FFNNNQSConfig
    from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    from src.training.vmc_trainer import ComplexVMCTrainer, VMCConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    if verbose:
        print("Training ComplexFFNNNQS...")
        print(f"  Model: hidden={hidden_size}, layers={n_layers}")
        print(f"  Training: epochs={n_epochs}, lr={learning_rate}")

    # Build Hamiltonian
    h2_config = H2Config(bond_length=0.74, bit_depth=n_qubits)
    hamiltonian = build_h2_hamiltonian_12bit(h2_config)
    electronic_fci = hamiltonian.electronic_fci_energy

    # Create model
    model_config = FFNNNQSConfig(
        n_visible=n_qubits,
        n_hidden=hidden_size,
        n_layers=n_layers,
    )
    model = ComplexFFNNNQS(model_config).to(device)

    # Configure VMC with particle constraint
    vmc_config = VMCConfig(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        use_exact=True,
        scheduler_type="cosine",
        warmup_epochs=max(50, n_epochs // 10),
        clip_grad_norm=1.0,
        use_particle_constraint=True,
        n_particles=n_particles,
    )

    # Train (suppress progress output)
    start_time = time.time()
    trainer = ComplexVMCTrainer(model, hamiltonian, vmc_config)

    with suppress_output():
        history = trainer.train()

    train_time = time.time() - start_time

    final_energy = history["energy"][-1]
    error_mha = abs(final_energy - electronic_fci) * 1000

    if verbose:
        print(f"  Training complete in {train_time:.1f}s")
        print(f"  VMC error: {error_mha:.3f} mHa")
        chem_acc = "YES" if error_mha < 1.6 else "NO"
        print(f"  Chemical accuracy: {chem_acc}")

    return model, error_mha


def run_sqd_comparison(
    n_sample_counts: List[int] = [100, 300, 500, 1000, 2000],
    n_qubits: int = 4,
    n_electrons: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> List[SQDResult]:
    """Run SQD comparison between random and NQS sampling.

    Parameters
    ----------
    n_sample_counts:
        List of sample counts to test.
    n_qubits:
        Number of qubits (4 for minimal H₂).
    n_electrons:
        Number of electrons (2 for H₂).
    seed:
        Random seed for reproducibility.
    verbose:
        Print progress information.

    Returns
    -------
    results:
        List of SQDResult objects for each sampler/sample_count combination.
    """
    from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    from src.sqd_interface.sqd_runner import run_sqd_on_samples
    from src.sqd_interface.sampling_adapters import (
        BaselineBernoulliSampler,
        ComplexFFNNSampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build Hamiltonian
    h2_config = H2Config(bond_length=0.74, bit_depth=n_qubits)
    hamiltonian = build_h2_hamiltonian_12bit(h2_config)
    electronic_fci = hamiltonian.electronic_fci_energy

    if verbose:
        print(f"\n{'='*70}")
        print("SQD Integration Test: Random vs NQS Sampling")
        print(f"{'='*70}")
        print(f"System: {n_qubits}-qubit H₂")
        print(f"Electronic FCI energy: {electronic_fci:.6f} Ha")
        print(f"Sample counts to test: {n_sample_counts}")

    # Train NQS model
    if verbose:
        print(f"\n{'='*70}")
        print("Step 1: Train ComplexFFNNNQS")
        print(f"{'='*70}")

    nqs_model, nqs_vmc_error = train_complex_nqs(
        n_qubits=n_qubits,
        n_epochs=1000,  # Enough for good convergence
        hidden_size=32,
        n_layers=2,
        learning_rate=5e-3,
        n_particles=n_electrons,
        verbose=verbose,
    )

    # Create samplers
    random_sampler = BaselineBernoulliSampler(n_visible=n_qubits)
    # Use (n_α, n_β) filtering for NQS sampler to match physical spin sector
    nqs_sampler = ComplexFFNNSampler(
        model=nqs_model,
        device=device,
        n_alpha=n_electrons // 2,
        n_beta=n_electrons // 2,
    )

    if verbose:
        print(f"\n{'='*70}")
        print("Step 2: Run SQD with Different Samplers")
        print(f"{'='*70}")

    results = []

    for n_samples in n_sample_counts:
        if verbose:
            print(f"\n--- Testing with {n_samples} samples ---")

        # Test random sampler with postselection
        if verbose:
            print("  Random sampler...")
        start_time = time.time()

        # Generate random samples with correct (n_α, n_β) spin structure
        # For H₂ singlet: (1α, 1β)
        n_alpha = n_electrons // 2
        n_beta = n_electrons // 2
        random_samples = generate_valid_random_samples(
            n_samples=n_samples,
            n_qubits=n_qubits,
            n_alpha=n_alpha,
            n_beta=n_beta,
            seed=seed,
        )

        try:
            sqd_result_random = run_sqd_on_samples(
                hamiltonian=hamiltonian,
                samples=random_samples,
                samples_per_batch=min(100, n_samples // 2),
                num_batches=5,
                max_iterations=3,
                seed=seed,
            )
            random_error = abs(sqd_result_random["electronic_energy"] - electronic_fci) * 1000
            random_time = time.time() - start_time

            results.append(SQDResult(
                sampler_name="Random",
                n_samples=n_samples,
                energy_estimate=sqd_result_random["energy_estimate"],
                electronic_energy=sqd_result_random["electronic_energy"],
                fci_energy=electronic_fci,
                error_mha=random_error,
                n_unique_samples=sqd_result_random["n_unique_samples"],
                subspace_dim=sqd_result_random["subspace_dim"],
                time_seconds=random_time,
            ))

            if verbose:
                print(f"    Error: {random_error:.3f} mHa, unique: {sqd_result_random['n_unique_samples']}")
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")

        # Test NQS sampler
        if verbose:
            print("  NQS sampler...")
        start_time = time.time()

        nqs_samples = nqs_sampler.sample(n_samples=n_samples)

        try:
            sqd_result_nqs = run_sqd_on_samples(
                hamiltonian=hamiltonian,
                samples=nqs_samples,
                samples_per_batch=min(100, n_samples // 2),
                num_batches=5,
                max_iterations=3,
                seed=seed,
            )
            nqs_error = abs(sqd_result_nqs["electronic_energy"] - electronic_fci) * 1000
            nqs_time = time.time() - start_time

            results.append(SQDResult(
                sampler_name="NQS",
                n_samples=n_samples,
                energy_estimate=sqd_result_nqs["energy_estimate"],
                electronic_energy=sqd_result_nqs["electronic_energy"],
                fci_energy=electronic_fci,
                error_mha=nqs_error,
                n_unique_samples=sqd_result_nqs["n_unique_samples"],
                subspace_dim=sqd_result_nqs["subspace_dim"],
                time_seconds=nqs_time,
            ))

            if verbose:
                print(f"    Error: {nqs_error:.3f} mHa, unique: {sqd_result_nqs['n_unique_samples']}")
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")

    return results


def generate_valid_random_samples(
    n_samples: int,
    n_qubits: int,
    n_alpha: int,
    n_beta: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate random samples with correct (n_α, n_β) electron numbers.

    For physical fermionic systems like H₂, we need to enforce both:
    - Total particle number: n_α + n_β
    - Spin sector: exactly n_α electrons in α spin-orbitals, n_β in β spin-orbitals

    In the Jordan-Wigner encoding with interleaved spin-orbitals:
    - Even indices (0, 2, 4, ...) are α spin-orbitals
    - Odd indices (1, 3, 5, ...) are β spin-orbitals

    For 4-qubit H₂ with (1α, 1β), valid configurations are:
    |1100⟩, |1001⟩, |0110⟩, |0011⟩
    """
    from itertools import combinations
    np.random.seed(seed)

    n_spatial = n_qubits // 2
    alpha_indices = list(range(0, n_qubits, 2))  # [0, 2, ...]
    beta_indices = list(range(1, n_qubits, 2))   # [1, 3, ...]

    # Generate all valid (n_α, n_β) configurations
    all_valid = []
    for alpha_orbs in combinations(range(n_spatial), n_alpha):
        for beta_orbs in combinations(range(n_spatial), n_beta):
            config = [0] * n_qubits
            for orb in alpha_orbs:
                config[2 * orb] = 1  # α in even indices
            for orb in beta_orbs:
                config[2 * orb + 1] = 1  # β in odd indices
            all_valid.append(config)

    all_valid = np.array(all_valid, dtype=np.int8)
    n_valid = len(all_valid)

    # Sample with replacement from valid configurations
    indices = np.random.choice(n_valid, size=n_samples, replace=True)
    return all_valid[indices]


def print_summary(results: List[SQDResult]) -> None:
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY: SQD Sample Efficiency Comparison")
    print(f"{'='*80}")
    print(f"{'Sampler':<10} {'Samples':<10} {'Error (mHa)':<15} {'Unique':<10} {'Subspace':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r.sampler_name:<10} {r.n_samples:<10} {r.error_mha:<15.3f} {r.n_unique_samples:<10} {r.subspace_dim:<10}")

    print("=" * 80)

    # Analyze sample efficiency
    print("\nSample Efficiency Analysis:")
    print("-" * 40)

    # Group by sample count
    sample_counts = sorted(set(r.n_samples for r in results))

    for n_samples in sample_counts:
        random_results = [r for r in results if r.sampler_name == "Random" and r.n_samples == n_samples]
        nqs_results = [r for r in results if r.sampler_name == "NQS" and r.n_samples == n_samples]

        if random_results and nqs_results:
            random_error = random_results[0].error_mha
            nqs_error = nqs_results[0].error_mha
            improvement = random_error - nqs_error
            improvement_pct = (improvement / random_error) * 100 if random_error > 0 else 0

            print(f"  {n_samples} samples: Random={random_error:.3f} mHa, NQS={nqs_error:.3f} mHa")
            if improvement > 0:
                print(f"    → NQS {improvement_pct:.1f}% better ({improvement:.3f} mHa improvement)")
            else:
                print(f"    → Random {-improvement_pct:.1f}% better")


def main():
    """Run the NQS-SQD integration test."""
    print("=" * 70)
    print("NQS-SQD Integration Test")
    print("=" * 70)
    print("\nObjective: Compare sample efficiency of random vs NQS sampling for SQD")
    print("Hypothesis: Trained NQS should require fewer samples for same accuracy")

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nWARNING: No GPU available, using CPU")

    # Run comparison with various sample counts
    results = run_sqd_comparison(
        n_sample_counts=[100, 300, 500, 1000],
        n_qubits=4,
        n_electrons=2,
        seed=42,
        verbose=True,
    )

    # Print summary
    print_summary(results)

    # Final analysis
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    # Check if NQS consistently outperformed random
    sample_counts = sorted(set(r.n_samples for r in results))
    nqs_wins = 0
    total_comparisons = 0

    for n_samples in sample_counts:
        random_results = [r for r in results if r.sampler_name == "Random" and r.n_samples == n_samples]
        nqs_results = [r for r in results if r.sampler_name == "NQS" and r.n_samples == n_samples]

        if random_results and nqs_results:
            total_comparisons += 1
            if nqs_results[0].error_mha < random_results[0].error_mha:
                nqs_wins += 1

    if total_comparisons > 0:
        win_rate = nqs_wins / total_comparisons * 100
        print(f"\nNQS outperformed random sampling in {nqs_wins}/{total_comparisons} cases ({win_rate:.0f}%)")

        if win_rate > 50:
            print("\n✓ NQS sampling shows improved sample efficiency for SQD!")
        else:
            print("\nNote: NQS did not consistently outperform random sampling.")
            print("This may be due to:")
            print("  1. Small system size (4 qubits has only 6 valid configurations)")
            print("  2. SQD subspace dimension is already maximal")
            print("  3. Need more samples to see the benefit")

    return results


if __name__ == "__main__":
    main()
