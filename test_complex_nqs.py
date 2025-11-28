#!/usr/bin/env python
"""Test complex-valued NQS vs real-valued NQS for VMC training.

This script tests whether complex wavefunctions can solve the "sign problem"
and achieve better accuracy than real-valued NQS.

Hypothesis: Real NQS cannot represent the H₂ ground state sign structure:
    |ψ_GS⟩ ≈ 0.99|1100⟩ - 0.14|0011⟩  (note the negative sign!)

Real NQS: ψ(x) = exp(f(x)) > 0 always (cannot represent negative amplitudes)
Complex NQS: ψ(x) = |ψ(x)| * exp(i*φ(x)) can have negative via φ = π

Expected result: Complex NQS should achieve much better accuracy than 20.5 mHa.

Usage:
    python test_complex_nqs.py
"""

import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

# Enable TF32 for faster matmul on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class Result:
    """Container for experiment results."""
    model_type: str
    final_energy: float
    fci_energy: float
    error_mha: float
    training_time: float
    chemical_accuracy: bool


def run_real_nqs_test(
    n_qubits: int = 4,
    n_epochs: int = 300,
    hidden_size: int = 64,
    n_layers: int = 2,
    learning_rate: float = 5e-3,
) -> Result:
    """Run VMC test with real-valued NQS."""
    from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
    from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    from src.training.vmc_trainer import VMCTrainer, VMCConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Testing: Real-valued NQS (FFNNNQS)")
    print("=" * 60)

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
    model = FFNNNQS(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: FFNNNQS({hidden_size}, {n_layers} layers)")
    print(f"Parameters: {n_params:,}")
    print(f"Wavefunction type: Real (ψ > 0)")

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
    history = trainer.train()
    training_time = time.time() - start_time

    final_energy = history["energy"][-1]
    error = abs(final_energy - electronic_fci)
    error_mha = error * 1000
    chemical_accuracy = error < 0.0016

    print(f"\nFinal error: {error_mha:.3f} mHa")
    print(f"Chemical accuracy: {'YES' if chemical_accuracy else 'NO'}")

    return Result(
        model_type="Real NQS (FFNNNQS)",
        final_energy=final_energy,
        fci_energy=electronic_fci,
        error_mha=error_mha,
        training_time=training_time,
        chemical_accuracy=chemical_accuracy,
    )


def run_complex_nqs_test(
    n_qubits: int = 4,
    n_epochs: int = 300,
    hidden_size: int = 64,
    n_layers: int = 2,
    learning_rate: float = 5e-3,
) -> Result:
    """Run VMC test with complex-valued NQS."""
    from src.nqs_models.ffn_nqs import ComplexFFNNNQS, FFNNNQSConfig
    from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    from src.training.vmc_trainer import ComplexVMCTrainer, VMCConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Testing: Complex-valued NQS (ComplexFFNNNQS)")
    print("=" * 60)

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ComplexFFNNNQS({hidden_size}, {n_layers} layers)")
    print(f"Parameters: {n_params:,} (2x real: amplitude + phase networks)")
    print(f"Wavefunction type: Complex (ψ = |ψ| * exp(i*φ))")

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
    trainer = ComplexVMCTrainer(model, hamiltonian, vmc_config)
    history = trainer.train()
    training_time = time.time() - start_time

    final_energy = history["energy"][-1]
    error = abs(final_energy - electronic_fci)
    error_mha = error * 1000
    chemical_accuracy = error < 0.0016

    print(f"\nFinal error: {error_mha:.3f} mHa")
    print(f"Chemical accuracy: {'YES' if chemical_accuracy else 'NO'}")

    # Analyze learned phases
    print("\nAnalyzing learned wavefunction phases...")
    analyze_learned_phases(model, n_qubits, device)

    return Result(
        model_type="Complex NQS (ComplexFFNNNQS)",
        final_energy=final_energy,
        fci_energy=electronic_fci,
        error_mha=error_mha,
        training_time=training_time,
        chemical_accuracy=chemical_accuracy,
    )


def analyze_learned_phases(model, n_qubits: int, device):
    """Analyze the learned phases of the complex NQS."""
    from src.training.vmc_trainer import enumerate_configurations

    all_configs = enumerate_configurations(n_qubits, device)

    with torch.no_grad():
        log_amp, phase = model(all_configs)

        # Compute normalized amplitudes
        log_prob = 2.0 * log_amp
        prob = torch.exp(log_prob - torch.logsumexp(log_prob, dim=0))
        amp = torch.sqrt(prob)

        # Convert phase to effective sign: cos(φ) ≈ +1 or -1
        effective_sign = torch.cos(phase)

        print(f"\n{'Config':<10} {'|ψ|':<12} {'Phase (rad)':<12} {'cos(φ)':<10} {'Effective Sign':<15}")
        print("-" * 60)

        for i in range(len(all_configs)):
            config_str = "".join([str(int(c)) for c in all_configs[i].tolist()])
            amp_val = amp[i].item()
            phase_val = phase[i].item()
            cos_val = effective_sign[i].item()

            # Only show significant amplitudes
            if amp_val > 0.01:
                sign_str = "+" if cos_val > 0 else "-"
                print(f"|{config_str}⟩    {amp_val:<12.4f} {phase_val:<12.4f} {cos_val:<10.4f} {sign_str:<15}")

        print("-" * 60)
        print("\nExpected for H₂ ground state:")
        print("  |1100⟩: large amplitude, φ ≈ 0 (positive)")
        print("  |0011⟩: smaller amplitude, φ ≈ π (negative)")


def main():
    """Run comparison experiment."""
    print("=" * 70)
    print("Complex vs Real NQS Comparison for VMC")
    print("=" * 70)
    print("\nHypothesis: Complex NQS can represent sign structure that real NQS cannot")
    print("Expected: Complex NQS achieves chemical accuracy, real NQS stuck at ~20 mHa")

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nWARNING: No GPU available, using CPU")

    results = []

    # Test 1: Real NQS (baseline)
    result_real = run_real_nqs_test(
        n_qubits=4,
        n_epochs=300,
        hidden_size=64,
        n_layers=2,
        learning_rate=5e-3,
    )
    results.append(result_real)

    # Test 2: Complex NQS
    result_complex = run_complex_nqs_test(
        n_qubits=4,
        n_epochs=300,
        hidden_size=64,
        n_layers=2,
        learning_rate=5e-3,
    )
    results.append(result_complex)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Complex vs Real NQS")
    print("=" * 70)
    print(f"{'Model Type':<30} {'Error (mHa)':<15} {'Chem Acc':<12} {'Time (s)':<10}")
    print("-" * 70)
    for r in results:
        status = "YES" if r.chemical_accuracy else "NO"
        print(f"{r.model_type:<30} {r.error_mha:<15.3f} {status:<12} {r.training_time:<10.1f}")
    print("=" * 70)

    # Analysis
    improvement = result_real.error_mha - result_complex.error_mha
    improvement_pct = (improvement / result_real.error_mha) * 100

    print(f"\nImprovement from complex NQS: {improvement:.3f} mHa ({improvement_pct:.1f}%)")

    if result_complex.chemical_accuracy:
        print("\n*** SUCCESS: Chemical accuracy achieved with complex NQS! ***")
        print("The sign problem was indeed the bottleneck.")
    elif result_complex.error_mha < result_real.error_mha * 0.5:
        print(f"\n** Significant improvement with complex NQS **")
        print("Sign problem partially addressed, but optimization may need tuning.")
    else:
        print("\nComplex NQS did not significantly improve accuracy.")
        print("Possible issues:")
        print("  1. Phase network optimization is difficult")
        print("  2. Need different hyperparameters for complex training")
        print("  3. May need longer training or different architecture")

    return results


if __name__ == "__main__":
    main()
