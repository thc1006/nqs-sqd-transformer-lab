"""Ablation driver for NQS vs baseline sampler experiments.

This module provides a unified experiment driver that:
1. Trains FFNN and Transformer NQS models using VMC
2. Compares sample efficiency across different samplers (baseline, FFNN, Transformer)
3. Runs SQD on samples from each sampler type
4. Saves results and generates comparison plots

The key research question: how much does a better sampler reduce sample complexity
for reaching accurate ground-state energies?
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
from src.nqs_models.transformer_nqs import TransformerNQS, TransformerNQSConfig
from src.nqs_models.utils import count_parameters
from src.sqd_interface.hamiltonian import H2Config, H2Hamiltonian, build_h2_hamiltonian_12bit
from src.sqd_interface.sampling_adapters import (
    BaselineBernoulliSampler,
    FFNNSampler,
    TransformerSampler,
)
from src.sqd_interface.sqd_runner import run_sqd_on_samples
from src.training.vmc_trainer import VMCConfig, VMCTrainer, compute_energy_exact


@dataclass
class AblationConfig:
    """Configuration for ablation experiments.

    Attributes
    ----------
    molecule_name:
        Name of the molecule (e.g., "H2").
    bond_length:
        Bond length in Angstrom.
    bit_depth:
        Number of qubits for encoding.
    samplers:
        List of sampler types to compare: "baseline", "ffnn", "transformer".
    sample_counts:
        List of sample counts to test for sample complexity.
    n_repeats:
        Number of repetitions per configuration for statistical analysis.
    seed:
        Random seed for reproducibility.
    output_dir:
        Directory to save results.
    ffnn_config:
        Configuration for FFNN NQS.
    transformer_config:
        Configuration for Transformer NQS.
    vmc_config:
        Configuration for VMC training.
    sqd_config:
        Configuration for SQD.
    """

    molecule_name: str = "H2"
    bond_length: float = 0.74
    bit_depth: int = 4
    samplers: List[str] = None
    sample_counts: List[int] = None
    n_repeats: int = 3
    seed: int = 42
    output_dir: str = "results/ablation"
    ffnn_config: Dict[str, Any] = None
    transformer_config: Dict[str, Any] = None
    vmc_config: Dict[str, Any] = None
    sqd_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.samplers is None:
            self.samplers = ["baseline", "ffnn", "transformer"]
        if self.sample_counts is None:
            self.sample_counts = [50, 100, 200, 500, 1000]
        if self.ffnn_config is None:
            self.ffnn_config = {"n_hidden": 64, "n_layers": 2}
        if self.transformer_config is None:
            self.transformer_config = {"d_model": 64, "n_heads": 4, "n_layers": 2}
        if self.vmc_config is None:
            self.vmc_config = {
                "learning_rate": 5e-3,
                "n_epochs": 100,
                "use_exact": True,
                "optimizer_type": "adam",
                "scheduler_type": "cosine",
                "warmup_epochs": 10,
                "min_lr": 1e-6,
            }
        if self.sqd_config is None:
            self.sqd_config = {
                "samples_per_batch": 100,
                "num_batches": 5,
                "max_iterations": 3,
            }


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""

    sampler: str
    n_samples: int
    repeat_idx: int
    energy_estimate: float
    electronic_energy: float
    error_vs_fci: float
    subspace_dim: int
    n_unique_samples: int
    converged: bool
    vmc_energy: Optional[float] = None
    vmc_error: Optional[float] = None


def load_config(path: str | pathlib.Path) -> AblationConfig:
    """Load configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    return AblationConfig(
        molecule_name=cfg_dict.get("molecule", {}).get("name", "H2"),
        bond_length=cfg_dict.get("molecule", {}).get("bond_length", 0.74),
        bit_depth=cfg_dict.get("encoding", {}).get("bits", 4),
        samplers=cfg_dict.get("ablation", {}).get("samplers", ["baseline", "ffnn", "transformer"]),
        sample_counts=cfg_dict.get("ablation", {}).get("sample_counts", [50, 100, 200, 500, 1000]),
        n_repeats=cfg_dict.get("ablation", {}).get("n_repeats", 3),
        seed=cfg_dict.get("seed", 42),
        output_dir=cfg_dict.get("output_dir", "results/ablation"),
        ffnn_config=cfg_dict.get("ffnn", {"n_hidden": 64, "n_layers": 2}),
        transformer_config=cfg_dict.get("transformer", {"d_model": 64, "n_heads": 4, "n_layers": 2}),
        vmc_config=cfg_dict.get("vmc", {"learning_rate": 5e-3, "n_epochs": 100, "use_exact": True}),
        sqd_config=cfg_dict.get("sqd", {"samples_per_batch": 100, "num_batches": 5, "max_iterations": 3}),
    )


def train_ffnn_nqs(
    hamiltonian: H2Hamiltonian,
    config: FFNNNQSConfig,
    vmc_config: VMCConfig,
    device: torch.device,
) -> FFNNNQS:
    """Train an FFNN NQS model using VMC."""
    model = FFNNNQS(config).to(device)
    trainer = VMCTrainer(model, hamiltonian, vmc_config)
    trainer.train()
    return model


def train_transformer_nqs(
    hamiltonian: H2Hamiltonian,
    config: TransformerNQSConfig,
    vmc_config: VMCConfig,
    device: torch.device,
) -> TransformerNQS:
    """Train a Transformer NQS model using VMC.

    Uses the TransformerVMCTrainer which properly handles the autoregressive
    probability factorization for VMC training.
    """
    from src.training.vmc_trainer import TransformerVMCTrainer

    model = TransformerNQS(config).to(device)
    trainer = TransformerVMCTrainer(model, hamiltonian, vmc_config)
    trainer.train()
    return model


def run_single_experiment(
    sampler_type: str,
    n_samples: int,
    hamiltonian: H2Hamiltonian,
    ffnn_model: Optional[FFNNNQS],
    transformer_model: Optional[TransformerNQS],
    sqd_config: Dict[str, Any],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Run a single SQD experiment with the specified sampler."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_visible = hamiltonian.num_qubits
    n_electrons = hamiltonian.num_electrons

    # Generate samples based on sampler type
    if sampler_type == "baseline":
        sampler = BaselineBernoulliSampler(n_visible=n_visible)
        samples = sampler.sample(n_samples=n_samples)
        # Postselect baseline samples to correct particle number
        particle_counts = samples.sum(axis=1)
        valid_mask = particle_counts == n_electrons
        samples = samples[valid_mask]
        if len(samples) < 10:
            # Not enough valid samples, generate more
            samples = sampler.sample(n_samples=n_samples * 20)
            particle_counts = samples.sum(axis=1)
            valid_mask = particle_counts == n_electrons
            samples = samples[valid_mask][:n_samples]
    elif sampler_type == "ffnn":
        if ffnn_model is None:
            raise ValueError("FFNN model not provided")
        sampler = FFNNSampler(model=ffnn_model, device=device, n_electrons=n_electrons)
        samples = sampler.sample(n_samples=n_samples)
    elif sampler_type == "transformer":
        if transformer_model is None:
            raise ValueError("Transformer model not provided")
        sampler = TransformerSampler(model=transformer_model, device=device, n_electrons=n_electrons)
        samples = sampler.sample(n_samples=n_samples)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    # Run SQD
    result = run_sqd_on_samples(
        hamiltonian=hamiltonian,
        samples=samples,
        samples_per_batch=sqd_config.get("samples_per_batch", 100),
        num_batches=sqd_config.get("num_batches", 5),
        max_iterations=sqd_config.get("max_iterations", 3),
        seed=seed,
    )

    # Compute error vs FCI
    error_vs_fci = abs(result["energy_estimate"] - hamiltonian.fci_energy)

    return {
        "energy_estimate": result["energy_estimate"],
        "electronic_energy": result["electronic_energy"],
        "error_vs_fci": error_vs_fci,
        "subspace_dim": result["subspace_dim"],
        "n_unique_samples": result["n_unique_samples"],
        "converged": result["converged"],
    }


def run_ablation(config: AblationConfig, verbose: bool = True) -> pd.DataFrame:
    """Run the full ablation study.

    Parameters
    ----------
    config:
        Ablation configuration.
    verbose:
        Whether to print progress information.

    Returns
    -------
    results_df:
        DataFrame with all experiment results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if verbose:
        print(f"Running ablation study on {device}")
        print(f"Molecule: {config.molecule_name}, bond length: {config.bond_length} A")
        print(f"Bit depth: {config.bit_depth}")
        print(f"Samplers: {config.samplers}")
        print(f"Sample counts: {config.sample_counts}")
        print(f"Repeats per config: {config.n_repeats}")

    # Build Hamiltonian
    h2_cfg = H2Config(bond_length=config.bond_length, bit_depth=config.bit_depth)
    hamiltonian = build_h2_hamiltonian_12bit(h2_cfg)

    if verbose:
        print(f"\nFCI reference energy: {hamiltonian.fci_energy:.6f} Ha")
        print(f"Nuclear repulsion: {hamiltonian.nuclear_repulsion:.6f} Ha")

    # Train NQS models if needed
    ffnn_model = None
    transformer_model = None
    vmc_energies = {}

    if "ffnn" in config.samplers:
        if verbose:
            print("\nTraining FFNN NQS model...")

        ffnn_cfg = FFNNNQSConfig(
            n_visible=config.bit_depth,
            n_hidden=config.ffnn_config.get("n_hidden", 64),
            n_layers=config.ffnn_config.get("n_layers", 2),
        )
        vmc_cfg = VMCConfig(
            learning_rate=config.vmc_config.get("learning_rate", 5e-3),
            n_epochs=config.vmc_config.get("n_epochs", 100),
            use_exact=config.vmc_config.get("use_exact", True),
            optimizer_type=config.vmc_config.get("optimizer_type", "adam"),
            scheduler_type=config.vmc_config.get("scheduler_type", "cosine"),
            warmup_epochs=config.vmc_config.get("warmup_epochs", 10),
            min_lr=config.vmc_config.get("min_lr", 1e-6),
            sr_regularization=config.vmc_config.get("sr_regularization", 0.01),
        )
        ffnn_model = train_ffnn_nqs(hamiltonian, ffnn_cfg, vmc_cfg, device)
        vmc_energy = compute_energy_exact(ffnn_model, hamiltonian, device)
        vmc_energies["ffnn"] = vmc_energy

        if verbose:
            print(f"FFNN VMC energy: {vmc_energy:.6f} Ha")
            print(f"FFNN VMC error: {abs(vmc_energy - hamiltonian.fci_energy):.6f} Ha")
            print(f"FFNN parameters: {count_parameters(ffnn_model)}")

    if "transformer" in config.samplers:
        if verbose:
            print("\nTraining Transformer NQS model...")

        transformer_cfg = TransformerNQSConfig(
            n_visible=config.bit_depth,
            d_model=config.transformer_config.get("d_model", 64),
            n_heads=config.transformer_config.get("n_heads", 4),
            n_layers=config.transformer_config.get("n_layers", 2),
        )
        vmc_cfg = VMCConfig(
            learning_rate=config.vmc_config.get("learning_rate", 5e-3),
            n_epochs=config.vmc_config.get("n_epochs", 100),
            use_exact=config.vmc_config.get("use_exact", True),
            optimizer_type=config.vmc_config.get("optimizer_type", "adam"),
            scheduler_type=config.vmc_config.get("scheduler_type", "cosine"),
            warmup_epochs=config.vmc_config.get("warmup_epochs", 10),
            min_lr=config.vmc_config.get("min_lr", 1e-6),
            sr_regularization=config.vmc_config.get("sr_regularization", 0.01),
        )
        transformer_model = train_transformer_nqs(hamiltonian, transformer_cfg, vmc_cfg, device)
        vmc_energy = compute_energy_exact(transformer_model, hamiltonian, device)
        vmc_energies["transformer"] = vmc_energy

        if verbose:
            print(f"Transformer VMC energy: {vmc_energy:.6f} Ha")
            print(f"Transformer VMC error: {abs(vmc_energy - hamiltonian.fci_energy):.6f} Ha")
            print(f"Transformer parameters: {count_parameters(transformer_model)}")

    # Run experiments
    results = []
    total_experiments = len(config.samplers) * len(config.sample_counts) * config.n_repeats

    if verbose:
        print(f"\nRunning {total_experiments} experiments...")

    pbar = tqdm(total=total_experiments, desc="Ablation", disable=not verbose)

    for sampler_type in config.samplers:
        for n_samples in config.sample_counts:
            for repeat_idx in range(config.n_repeats):
                seed = config.seed + repeat_idx * 1000 + hash(sampler_type) % 100

                try:
                    exp_result = run_single_experiment(
                        sampler_type=sampler_type,
                        n_samples=n_samples,
                        hamiltonian=hamiltonian,
                        ffnn_model=ffnn_model,
                        transformer_model=transformer_model,
                        sqd_config=config.sqd_config,
                        device=device,
                        seed=seed,
                    )

                    result = ExperimentResult(
                        sampler=sampler_type,
                        n_samples=n_samples,
                        repeat_idx=repeat_idx,
                        energy_estimate=exp_result["energy_estimate"],
                        electronic_energy=exp_result["electronic_energy"],
                        error_vs_fci=exp_result["error_vs_fci"],
                        subspace_dim=exp_result["subspace_dim"],
                        n_unique_samples=exp_result["n_unique_samples"],
                        converged=exp_result["converged"],
                        vmc_energy=vmc_energies.get(sampler_type),
                        vmc_error=abs(vmc_energies.get(sampler_type, 0) - hamiltonian.fci_energy)
                        if sampler_type in vmc_energies
                        else None,
                    )
                    results.append(asdict(result))
                except Exception as e:
                    if verbose:
                        print(f"\nError in {sampler_type}, n={n_samples}, rep={repeat_idx}: {e}")
                    results.append({
                        "sampler": sampler_type,
                        "n_samples": n_samples,
                        "repeat_idx": repeat_idx,
                        "energy_estimate": np.nan,
                        "electronic_energy": np.nan,
                        "error_vs_fci": np.nan,
                        "subspace_dim": 0,
                        "n_unique_samples": 0,
                        "converged": False,
                        "vmc_energy": None,
                        "vmc_error": None,
                    })

                pbar.update(1)

    pbar.close()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Add metadata
    results_df.attrs["fci_energy"] = hamiltonian.fci_energy
    results_df.attrs["nuclear_repulsion"] = hamiltonian.nuclear_repulsion
    results_df.attrs["config"] = asdict(config) if hasattr(config, "__dataclass_fields__") else config.__dict__

    return results_df


def save_results(results_df: pd.DataFrame, output_dir: str, experiment_name: str) -> None:
    """Save results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(output_dir, f"{experiment_name}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Save metadata as JSON
    metadata = {
        "fci_energy": results_df.attrs.get("fci_energy"),
        "nuclear_repulsion": results_df.attrs.get("nuclear_repulsion"),
        "config": results_df.attrs.get("config"),
        "timestamp": datetime.now().isoformat(),
    }
    json_path = os.path.join(output_dir, f"{experiment_name}_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {json_path}")


def plot_sample_complexity(results_df: pd.DataFrame, output_dir: str, experiment_name: str) -> None:
    """Generate sample complexity plot comparing samplers."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Group by sampler and n_samples, compute mean and std
    grouped = results_df.groupby(["sampler", "n_samples"]).agg({
        "error_vs_fci": ["mean", "std"],
        "n_unique_samples": ["mean", "std"],
    }).reset_index()

    grouped.columns = ["sampler", "n_samples", "error_mean", "error_std", "unique_mean", "unique_std"]

    colors = {"baseline": "gray", "ffnn": "blue", "transformer": "red"}
    markers = {"baseline": "o", "ffnn": "s", "transformer": "^"}

    # Plot 1: Error vs sample count
    ax1 = axes[0]
    for sampler in grouped["sampler"].unique():
        data = grouped[grouped["sampler"] == sampler]
        ax1.errorbar(
            data["n_samples"],
            data["error_mean"],
            yerr=data["error_std"],
            label=sampler,
            color=colors.get(sampler, "black"),
            marker=markers.get(sampler, "o"),
            capsize=3,
        )

    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Error vs FCI (Ha)")
    ax1.set_title("Sample Complexity: Error vs Sample Count")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Unique samples vs total samples
    ax2 = axes[1]
    for sampler in grouped["sampler"].unique():
        data = grouped[grouped["sampler"] == sampler]
        ax2.errorbar(
            data["n_samples"],
            data["unique_mean"],
            yerr=data["unique_std"],
            label=sampler,
            color=colors.get(sampler, "black"),
            marker=markers.get(sampler, "o"),
            capsize=3,
        )

    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Unique Configurations")
    ax2.set_title("Sample Diversity: Unique vs Total Samples")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{experiment_name}_sample_complexity.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved plot to {fig_path}")


def print_summary(results_df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    fci = results_df.attrs.get("fci_energy", -1.137284)
    print(f"\nFCI Reference Energy: {fci:.6f} Ha")

    # Summary by sampler
    for sampler in results_df["sampler"].unique():
        data = results_df[results_df["sampler"] == sampler]
        print(f"\n{sampler.upper()} Sampler:")
        print(f"  Mean error: {data['error_vs_fci'].mean():.6f} +/- {data['error_vs_fci'].std():.6f} Ha")
        print(f"  Best error: {data['error_vs_fci'].min():.6f} Ha")
        print(f"  Convergence rate: {data['converged'].mean() * 100:.1f}%")

    # Best results per sample count
    print("\nBest Results by Sample Count:")
    for n in sorted(results_df["n_samples"].unique()):
        data = results_df[results_df["n_samples"] == n]
        best_idx = data["error_vs_fci"].idxmin()
        best = results_df.loc[best_idx]
        print(f"  n={n}: {best['sampler']} with error {best['error_vs_fci']:.6f} Ha")


def main() -> None:
    """Main entry point for ablation experiments."""
    parser = argparse.ArgumentParser(description="NQS vs Baseline Ablation Study")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=4,
        help="Number of qubits (bit depth).",
    )
    parser.add_argument(
        "--samplers",
        type=str,
        nargs="+",
        default=["baseline", "ffnn"],
        help="Samplers to compare.",
    )
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Sample counts to test.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of VMC training epochs.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sr", "sgd"],
        help="Optimizer type for VMC training.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step", "exponential"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment run.",
    )
    args = parser.parse_args()

    # Load config from file or create from args
    if args.config:
        config = load_config(args.config)
    else:
        config = AblationConfig(
            bit_depth=args.bit_depth,
            samplers=args.samplers,
            sample_counts=args.sample_counts,
            n_repeats=args.n_repeats,
            seed=args.seed,
            output_dir=args.output_dir,
            vmc_config={
                "learning_rate": 5e-3,
                "n_epochs": args.n_epochs,
                "use_exact": True,
                "optimizer_type": args.optimizer,
                "scheduler_type": args.scheduler,
                "warmup_epochs": 10,
                "min_lr": 1e-6,
                "sr_regularization": 0.01,
            },
        )

    # Generate experiment name
    experiment_name = args.experiment_name or f"ablation_{config.bit_depth}bit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run ablation
    results_df = run_ablation(config, verbose=True)

    # Save results
    save_results(results_df, config.output_dir, experiment_name)

    # Generate plots
    figures_dir = os.path.join(config.output_dir, "figures")
    plot_sample_complexity(results_df, figures_dir, experiment_name)

    # Print summary
    print_summary(results_df)

    print(f"\nExperiment complete! Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
