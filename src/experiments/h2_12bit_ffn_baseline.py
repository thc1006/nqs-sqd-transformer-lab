"""H2 @ 12-bit: FFNN NQS vs baseline sampler (stub)."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import torch
import yaml

from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
from src.nqs_models.utils import count_parameters
from src.sqd_interface.hamiltonian import H2Config, build_h2_hamiltonian_12bit
from src.sqd_interface.sampling_adapters import BaselineBernoulliSampler, FFNNSampler
from src.sqd_interface.sqd_runner import run_sqd_on_samples


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 12-bit FFNN NQS vs baseline (stub)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h2_12bit_ffn.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[INFO] Loaded config from {args.config}:")
    print(cfg)

    h2_cfg = H2Config(
        bond_length=cfg.get("molecule", {}).get("bond_length", 0.74),
        bit_depth=cfg.get("encoding", {}).get("bits", 12),
    )
    try:
        hamiltonian = build_h2_hamiltonian_12bit(h2_cfg)
    except NotImplementedError:
        hamiltonian = None
        print("[WARN] build_h2_hamiltonian_12bit is not implemented yet; using None as placeholder.")

    n_visible = cfg.get("encoding", {}).get("bits", 12)
    n_samples = cfg.get("sampling", {}).get("n_samples", 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline sampler
    baseline_sampler = BaselineBernoulliSampler(n_visible=n_visible)
    baseline_samples = baseline_sampler.sample(n_samples=n_samples)
    baseline_result = run_sqd_on_samples(
        hamiltonian=hamiltonian,
        samples=baseline_samples,
        max_subspace_dim=cfg.get("sqd", {}).get("max_subspace_dim", 256),
    )
    print("[RESULT] Baseline SQD result (stub):", baseline_result)

    # FFNN NQS sampler (untrained / placeholder)
    nqs_cfg = FFNNNQSConfig(
        n_visible=n_visible,
        n_hidden=cfg.get("nqs", {}).get("n_hidden", 64),
        n_layers=cfg.get("nqs", {}).get("n_layers", 2),
    )
    model = FFNNNQS(nqs_cfg).to(device)
    print(f"[INFO] FFNNNQS parameters: {count_parameters(model)}")

    nqs_sampler = FFNNSampler(model=model, device=device)
    nqs_samples = nqs_sampler.sample(n_samples=n_samples)
    nqs_result = run_sqd_on_samples(
        hamiltonian=hamiltonian,
        samples=nqs_samples,
        max_subspace_dim=cfg.get("sqd", {}).get("max_subspace_dim", 256),
    )
    print("[RESULT] FFNN NQS SQD result (stub):", nqs_result)
    print("[NOTE] Training loops and real SQD calls should be added with Claude Code.")


if __name__ == "__main__":
    main()
