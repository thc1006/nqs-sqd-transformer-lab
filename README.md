# NQS-guided SQD Lab (FFNN + Transformer)

This repo is a **playground for NQS-guided Sample-based Quantum Diagonalization (SQD)**,
targeting small molecular systems (starting from H₂) under 12–14-bit encodings.

The core idea:

- Use **Neural Quantum States (NQS)** (FFNN and Transformer-based) as classical samplers
  to generate bitstring configurations.
- Use **`qiskit-addon-sqd`** as a sample-based diagonalization backend to estimate
  ground-state energies in the subspace spanned by these configurations.
- Study **sample-efficiency**:
  - How many samples are required to reach near-FCI energies (e.g. ~ -7.63 Ha for H₂)?
  - How does FFNN vs Transformer NQS affect bias/variance and convergence?

This project is designed to be used with **Claude Code (Opus 4.5)**, Skills, and MCP.

For details on structure and workflows, see `CLAUDE.md`.

---

## Repacking / Exporting the Project

This repo includes helper scripts under `scripts/` to create a clean zip archive
after you modify the code (for sending to your PI, collaborators, or for backup).

### Linux / macOS / WSL

From the project root:

```bash
bash scripts/repack.sh
```

This will:

- create an `exports/` directory if it does not already exist, and
- write a timestamped zip like:

```text
exports/nqs-sqd-transformer-lab-YYYYMMDD_HHMMSS.zip
```

The script automatically excludes heavy / transient directories:

- `.venv/`
- `__pycache__/`
- `.git/`
- `exports/`
- `results/`
- `data/cached_samples/`

### Windows / Generic Python

From the project root, with your virtual environment activated:

```bash
python scripts/repack.py
```

This does the same thing as `repack.sh`, but purely in Python (no shell features).
