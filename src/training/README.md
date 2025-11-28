# VMC Training Module

This module implements Variational Monte Carlo (VMC) training for Neural Quantum States (NQS).

## Overview

The VMC trainer minimizes the energy functional:

```
E[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
```

For NQS models, this is computed via:

1. **Exact enumeration** (small systems): Enumerate all 2^n configurations and compute exact gradients
2. **Sampling** (large systems): Sample from |ψ(x)|² and estimate gradients via Monte Carlo

## Components

### `VMCConfig`
Configuration dataclass for training hyperparameters:
- `learning_rate`: Adam optimizer learning rate (default: 1e-3)
- `n_epochs`: Number of training epochs (default: 20)
- `n_samples`: Samples per gradient estimate for MC (default: 256)
- `batch_size`: Batch size for exact computation (default: 64)
- `use_exact`: Use exact enumeration vs sampling (default: True)
- `clip_grad_norm`: Gradient clipping threshold (default: 1.0)
- `weight_decay`: L2 regularization (default: 0.0)

### `VMCTrainer`
Main training class that supports:
- Both FFNN and Transformer NQS models
- Exact and sampling-based training modes
- GPU acceleration
- Progress tracking with tqdm
- Training history logging

### `compute_local_energy()`
Computes local energy E_loc(x) = ⟨x|H|ψ⟩ / ⟨x|ψ⟩ for a batch of configurations.

## Usage

### Basic Example

```python
import torch
from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
from src.training.vmc_trainer import VMCTrainer, VMCConfig

# Setup
device = torch.device("cuda")
hamiltonian = build_h2_hamiltonian_12bit(H2Config(bond_length=0.74, bit_depth=12))

# Create model
model_config = FFNNNQSConfig(n_visible=12, n_hidden=64, n_layers=2)
model = FFNNNQS(model_config).to(device)

# Configure training
vmc_config = VMCConfig(
    learning_rate=1e-3,
    n_epochs=50,
    use_exact=True,
)

# Train
trainer = VMCTrainer(model, hamiltonian, vmc_config)
history = trainer.train()

# Results
print(f"Final energy: {history['energy'][-1]:.6f} Ha")
print(f"FCI energy: {hamiltonian.fci_energy:.6f} Ha")
```

### Using Transformer NQS

```python
from src.nqs_models.transformer_nqs import TransformerNQS, TransformerNQSConfig

# Create Transformer model
model_config = TransformerNQSConfig(
    n_visible=12,
    d_model=128,
    n_heads=4,
    n_layers=2,
)
model = TransformerNQS(model_config).to(device)

# Train (same as FFNN)
trainer = VMCTrainer(model, hamiltonian, vmc_config)
history = trainer.train()
```

### Sampling-based Training (for larger systems)

```python
# For systems where exact enumeration is infeasible
vmc_config = VMCConfig(
    learning_rate=1e-3,
    n_epochs=100,
    n_samples=1000,  # Number of samples per gradient estimate
    use_exact=False,  # Use sampling instead of exact
)

trainer = VMCTrainer(model, hamiltonian, vmc_config)
history = trainer.train()
```

## Running Experiments

### Quick Test
```bash
python test_vmc_simple.py
```

This runs a minimal 10-epoch training on H2 (4 qubits) to verify the implementation.

### Full Experiment
```bash
python -m src.experiments.vmc_h2_baseline --config configs/vmc_training.yaml
```

This trains both FFNN and Transformer NQS on H2 (12 qubits) and generates comparison plots.

## Training History

The `train()` method returns a dictionary with:
- `energy`: Energy values per epoch
- `energy_std`: Energy standard deviation (for sampling mode)
- `loss`: Training loss values
- `grad_norm`: Gradient norms

## Implementation Notes

### Exact Mode (use_exact=True)
- Enumerates all 2^n configurations
- Computes exact energy and gradients
- Suitable for small systems (n ≤ 14 qubits)
- Zero sampling error

### Sampling Mode (use_exact=False)
- Samples from model distribution
- Estimates gradients via Monte Carlo
- Suitable for larger systems
- Has sampling error (controlled by n_samples)

### Local Energy Computation
The local energy is computed by:
1. Building the full Hamiltonian matrix (for small systems)
2. Computing the wavefunction for all configurations
3. Applying H|ψ⟩
4. Computing E_loc(x) = (H|ψ⟩)[x] / ψ(x)

For larger systems, you would implement sparse matrix operations or operator-specific routines.

## GPU Acceleration

The trainer automatically uses the device of the model:
- Hamiltonian matrices are moved to GPU
- All tensor operations run on GPU
- Batch processing for efficient computation

## Troubleshooting

### Energy not converging
- Increase `n_epochs`
- Decrease `learning_rate`
- Increase model capacity (n_hidden, n_layers)
- Check gradient norms (may need different clip_grad_norm)

### Out of memory
- Reduce `n_samples` (for sampling mode)
- Reduce `batch_size`
- Use sampling mode instead of exact for large systems

### Numerical instability
- Enable gradient clipping (clip_grad_norm=1.0)
- Add weight decay regularization
- Use smaller learning rate

## References

- Carleo & Troyer, "Solving the quantum many-body problem with artificial neural networks" (2017)
- McBrian et al., "Transformer Quantum State: A Multipurpose Model for Quantum Many-Body Problems" (2023)
