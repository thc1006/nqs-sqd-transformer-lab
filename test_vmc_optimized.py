#!/usr/bin/env python
"""Optimized VMC test with correct energy comparison and GPU optimizations.

This script:
1. Uses correct energy reference (electronic_fci_energy, not fci_energy)
2. Uses torch.compile for GPU acceleration
3. Tests both 4-bit and 12-bit systems
4. Reports results with proper precision metrics

Usage:
    python test_vmc_optimized.py
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
    system: str
    n_qubits: int
    electronic_fci: float
    final_energy: float
    error_mha: float
    training_time: float
    chemical_accuracy: bool


def run_vmc_test(
    n_qubits: int,
    n_epochs: int = 200,
    hidden_size: int = 64,
    n_layers: int = 2,
    learning_rate: float = 5e-3,
    use_compile: bool = True,
    verbose: bool = True,
) -> TestResult:
    """Run VMC training test with GPU optimizations.

    Parameters
    ----------
    n_qubits:
        Number of qubits (4 or 12 for H2)
    n_epochs:
        Number of training epochs
    hidden_size:
        FFNN hidden layer size
    n_layers:
        FFNN number of layers
    learning_rate:
        Optimizer learning rate
    use_compile:
        Whether to use torch.compile
    verbose:
        Print detailed output

    Returns
    -------
    TestResult with all metrics
    """
    from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
    from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    from src.training.vmc_trainer import VMCTrainer, VMCConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    if verbose:
        print(f"\n{'='*60}")
        print(f"VMC Test: H2 @ {n_qubits} qubits")
        print(f"{'='*60}")

    # Build Hamiltonian
    h2_config = H2Config(bond_length=0.74, bit_depth=n_qubits)
    hamiltonian = build_h2_hamiltonian_12bit(h2_config)

    # CRITICAL: Use electronic_fci_energy for comparison!
    electronic_fci = hamiltonian.electronic_fci_energy

    if verbose:
        print(f"\nEnergy Reference Points:")
        print(f"  Electronic FCI (H operator GS): {electronic_fci:.6f} Ha")
        print(f"  Nuclear repulsion:              {hamiltonian.nuclear_repulsion:.6f} Ha")
        print(f"  Total FCI (E_elec + E_nuc):     {hamiltonian.fci_energy:.6f} Ha")
        print(f"  Chemical accuracy threshold:    1.6 mHa = 0.0016 Ha")

    # Create model
    model_config = FFNNNQSConfig(
        n_visible=n_qubits,
        n_hidden=hidden_size,
        n_layers=n_layers,
    )
    model = FFNNNQS(model_config).to(device)

    # Optional: compile for speed (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if verbose:
                print(f"\n[GPU Optimization] torch.compile enabled")
        except Exception as e:
            if verbose:
                print(f"\n[GPU Optimization] torch.compile failed: {e}")

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\nModel: FFNNNQS({hidden_size}, {n_layers} layers)")
        print(f"Parameters: {n_params:,}")
        print(f"Device: {device}")

    # Configure VMC
    vmc_config = VMCConfig(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        use_exact=True,
        scheduler_type="cosine",
        warmup_epochs=max(10, n_epochs // 10),
        clip_grad_norm=1.0,
    )

    # Train
    start_time = time.time()
    trainer = VMCTrainer(model, hamiltonian, vmc_config)

    # Suppress tqdm for cleaner output if not verbose
    if not verbose:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    history = trainer.train()

    if not verbose:
        sys.stdout = old_stdout

    training_time = time.time() - start_time

    # Get final electronic energy from history
    final_electronic_energy = history["energy"][-1]

    # Calculate error vs electronic FCI (CORRECT comparison!)
    error = abs(final_electronic_energy - electronic_fci)
    error_mha = error * 1000  # Convert to mHa
    chemical_accuracy = error < 0.0016  # 1.6 mHa

    # Print results
    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS (Correct Energy Comparison)")
        print(f"{'='*60}")
        print(f"Final electronic energy:  {final_electronic_energy:.6f} Ha")
        print(f"Electronic FCI:           {electronic_fci:.6f} Ha")
        print(f"Error:                    {error:.6f} Ha ({error_mha:.3f} mHa)")
        print(f"Training time:            {training_time:.1f} s")
        print(f"Chemical accuracy (1.6 mHa): {'ACHIEVED' if chemical_accuracy else 'NOT achieved'}")
        print(f"{'='*60}")

        # Also show total energies for completeness
        final_total = final_electronic_energy + hamiltonian.nuclear_repulsion
        print(f"\n[Total energies for reference]")
        print(f"  Final total: {final_total:.6f} Ha")
        print(f"  FCI total:   {hamiltonian.fci_energy:.6f} Ha")

    return TestResult(
        system=f"H2_{n_qubits}bit",
        n_qubits=n_qubits,
        electronic_fci=electronic_fci,
        final_energy=final_electronic_energy,
        error_mha=error_mha,
        training_time=training_time,
        chemical_accuracy=chemical_accuracy,
    )


def main():
    """Run comprehensive VMC tests."""
    print("=" * 70)
    print("VMC Training Test with Correct Energy Reference & GPU Optimizations")
    print("=" * 70)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
    else:
        print("\nWARNING: No GPU available, using CPU")

    results = []

    # Test 1: 4-qubit system (fast, should converge well)
    print("\n" + "=" * 70)
    print("TEST 1: 4-qubit H2 (baseline)")
    print("=" * 70)
    result_4bit = run_vmc_test(
        n_qubits=4,
        n_epochs=300,
        hidden_size=64,
        n_layers=2,
        learning_rate=5e-3,
        use_compile=True,
        verbose=True,
    )
    results.append(result_4bit)

    # Test 2: 12-qubit system (larger, more challenging)
    print("\n" + "=" * 70)
    print("TEST 2: 12-qubit H2 (expanded)")
    print("=" * 70)
    result_12bit = run_vmc_test(
        n_qubits=12,
        n_epochs=500,
        hidden_size=128,
        n_layers=3,
        learning_rate=3e-3,
        use_compile=True,
        verbose=True,
    )
    results.append(result_12bit)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'System':<15} {'Qubits':<8} {'Error (mHa)':<12} {'Time (s)':<10} {'Chem. Acc.':<12}")
    print("-" * 70)
    for r in results:
        status = "YES" if r.chemical_accuracy else "NO"
        print(f"{r.system:<15} {r.n_qubits:<8} {r.error_mha:<12.3f} {r.training_time:<10.1f} {status:<12}")
    print("=" * 70)

    # Overall assessment
    all_passed = all(r.error_mha < 50 for r in results)  # 50 mHa threshold for "reasonable"
    if all_passed:
        print("\nAll tests passed with reasonable accuracy (<50 mHa)")
    else:
        print("\nSome tests have large errors - check training configuration")

    return results


if __name__ == "__main__":
    main()
