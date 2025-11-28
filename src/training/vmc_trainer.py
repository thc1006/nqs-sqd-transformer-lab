"""Variational Monte Carlo (VMC) trainer for Neural Quantum States.

This module implements VMC training for NQS models using both exact enumeration
(for small systems) and sampling-based approaches (for larger systems).

For small molecules like H₂ (4 spin orbitals, 16 configurations), we can:
1. Enumerate all configurations and compute exact gradients
2. Use sampling-based VMC for larger systems

The energy functional to minimize is:
    E[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ = Σ_x |ψ(x)|² E_loc(x)

where E_loc(x) = ⟨x|H|ψ⟩ / ⟨x|ψ⟩ is the local energy.

For real-valued NQS (log ψ real), the gradient is:
    ∇E = 2 * Σ_x |ψ(x)|² (E_loc(x) - E) * ∇log ψ(x)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..sqd_interface.hamiltonian import H2Hamiltonian
from .sr_optimizer import SROptimizer


@dataclass
class VMCConfig:
    """Configuration for VMC training.

    Attributes
    ----------
    learning_rate:
        Optimizer learning rate (η for SR, lr for Adam/SGD).
    n_epochs:
        Number of training epochs.
    n_samples:
        Number of samples per gradient estimate (for sampling-based VMC).
    batch_size:
        Batch size for gradient computation (for exact VMC).
    use_exact:
        If True, enumerate all configurations (only for small systems).
    clip_grad_norm:
        Maximum gradient norm for clipping (None for no clipping).
        Only applies to Adam/SGD, not SR.
    weight_decay:
        L2 regularization coefficient (only for Adam/SGD).
    scheduler_type:
        Learning rate scheduler type. Options: "none", "cosine", "step", "exponential".
        Note: Schedulers only apply to Adam/SGD, not SR.
    warmup_epochs:
        Number of epochs for linear warmup (used with cosine scheduler).
    min_lr:
        Minimum learning rate for schedulers.
    optimizer_type:
        Optimizer to use: "adam", "sr", or "sgd".
    sr_regularization:
        Regularization parameter ε for SR optimizer (S + ε*I).
    sr_use_iterative:
        If True, use conjugate gradient for SR instead of direct solve.
        Recommended for models with >10k parameters.
    """

    learning_rate: float = 1e-3
    n_epochs: int = 20
    n_samples: int = 256
    batch_size: int = 64
    use_exact: bool = True
    clip_grad_norm: Optional[float] = 1.0
    weight_decay: float = 0.0
    scheduler_type: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    optimizer_type: str = "adam"  # Options: "adam", "sr", "sgd"
    sr_regularization: float = 0.01
    sr_use_iterative: bool = False
    # Particle number constraint (粒子數約束)
    use_particle_constraint: bool = False
    n_particles: Optional[int] = None  # If None, use hamiltonian.num_electrons


def enumerate_configurations(n_qubits: int, device: torch.device) -> torch.Tensor:
    """Enumerate all 2^n binary configurations (GPU-optimized).

    Parameters
    ----------
    n_qubits:
        Number of qubits (visible units).
    device:
        Device to create tensor on.

    Returns
    -------
    configs:
        Tensor of shape (2^n_qubits, n_qubits) with all binary configurations.
    """
    n_configs = 2**n_qubits
    # Vectorized generation using bit manipulation (much faster than loop)
    indices = torch.arange(n_configs, device=device, dtype=torch.int64)
    # Extract each bit position
    configs = torch.zeros(n_configs, n_qubits, dtype=torch.float32, device=device)
    for j in range(n_qubits):
        configs[:, j] = ((indices >> (n_qubits - 1 - j)) & 1).float()
    return configs


def enumerate_configurations_fixed_n(
    n_qubits: int,
    n_particles: int,
    device: torch.device,
) -> torch.Tensor:
    """Enumerate configurations with fixed particle number (粒子數約束).

    This function generates only configurations where the number of 1s equals
    n_particles. For H₂ with 2 electrons in 4 qubits, this reduces the space
    from 2^4=16 to C(4,2)=6 configurations.

    Parameters
    ----------
    n_qubits:
        Number of qubits (visible units).
    n_particles:
        Fixed number of particles (number of 1s in each configuration).
    device:
        Device to create tensor on.

    Returns
    -------
    configs:
        Tensor of shape (C(n_qubits, n_particles), n_qubits) with valid configurations.

    Examples
    --------
    >>> configs = enumerate_configurations_fixed_n(4, 2, torch.device('cpu'))
    >>> configs.shape
    torch.Size([6, 4])
    >>> configs  # |1100⟩, |1010⟩, |1001⟩, |0110⟩, |0101⟩, |0011⟩
    tensor([[1., 1., 0., 0.],
            [1., 0., 1., 0.],
            [1., 0., 0., 1.],
            [0., 1., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 1.]])
    """
    from itertools import combinations

    # Generate all combinations of n_particles positions out of n_qubits
    valid_configs = []
    for positions in combinations(range(n_qubits), n_particles):
        config = [0.0] * n_qubits
        for pos in positions:
            config[pos] = 1.0
        valid_configs.append(config)

    configs = torch.tensor(valid_configs, dtype=torch.float32, device=device)
    return configs


def compute_local_energy_complex(
    model: nn.Module,
    hamiltonian: H2Hamiltonian,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute local energy for complex-valued NQS.

    E_loc(x) = ⟨x|H|ψ⟩ / ⟨x|ψ⟩

    For complex ψ(x) = |ψ(x)| * exp(i*φ(x)), the local energy can be complex
    for individual configurations, but E = Σ|ψ|²E_loc should be real.

    Parameters
    ----------
    model:
        Complex NQS model with forward() returning (log_amp, phase).
    hamiltonian:
        Hamiltonian object.
    x:
        Batch of configurations, shape (batch, n_qubits).

    Returns
    -------
    E_loc:
        Complex local energies, shape (batch,).
    """
    device = x.device
    n_qubits = x.shape[1]
    batch_size = x.shape[0]

    # Convert Hamiltonian to dense matrix
    ham_matrix = hamiltonian.hamiltonian.to_matrix()
    ham_matrix = torch.tensor(ham_matrix, dtype=torch.complex64, device=device)

    # Compute complex wavefunction for all configurations
    all_configs = enumerate_configurations(n_qubits, device)

    with torch.no_grad():
        log_amp, phase = model(all_configs)
        # ψ(x) = exp(log_amp + i*phase)
        psi_complex = torch.exp(torch.complex(log_amp, phase))

    # Normalize wavefunction (use |ψ|² norm)
    norm = torch.sqrt(torch.sum(torch.abs(psi_complex) ** 2))
    psi_complex = psi_complex / norm

    # Apply Hamiltonian: H|ψ⟩
    h_psi = ham_matrix @ psi_complex

    # Compute local energy for each input configuration
    E_loc = torch.zeros(batch_size, dtype=torch.complex64, device=device)

    for i in range(batch_size):
        config_idx = 0
        for j in range(n_qubits):
            if x[i, j] > 0.5:
                config_idx += 2 ** (n_qubits - 1 - j)

        psi_x = psi_complex[config_idx]
        h_psi_x = h_psi[config_idx]

        if torch.abs(psi_x) > 1e-10:
            E_loc[i] = h_psi_x / psi_x
        else:
            E_loc[i] = 0.0 + 0.0j

    return E_loc


def compute_energy_exact_complex(
    model: nn.Module,
    hamiltonian: H2Hamiltonian,
    device: torch.device,
) -> float:
    """Compute exact energy for complex-valued NQS.

    E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩

    Parameters
    ----------
    model:
        Complex NQS model with forward() returning (log_amp, phase).
    hamiltonian:
        Hamiltonian object.
    device:
        Device for computation.

    Returns
    -------
    energy:
        Energy expectation value (real).
    """
    n_qubits = hamiltonian.num_qubits
    all_configs = enumerate_configurations(n_qubits, device)

    with torch.no_grad():
        log_amp, phase = model(all_configs)
        psi_complex = torch.exp(torch.complex(log_amp, phase))
        norm_sq = torch.sum(torch.abs(psi_complex) ** 2)
        psi_normalized = psi_complex / torch.sqrt(norm_sq)

    # Compute ⟨ψ|H|ψ⟩
    ham_matrix = hamiltonian.hamiltonian.to_matrix()
    ham_matrix = torch.tensor(ham_matrix, dtype=torch.complex64, device=device)

    energy = torch.real(psi_normalized.conj() @ ham_matrix @ psi_normalized)
    return energy.item()


def compute_local_energy(
    model: nn.Module,
    hamiltonian: H2Hamiltonian,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute local energy E_loc(x) = ⟨x|H|ψ⟩ / ⟨x|ψ⟩.

    For a given configuration x, the local energy is computed by applying
    the Hamiltonian to the state |ψ⟩ and dividing by the wavefunction amplitude.

    Parameters
    ----------
    model:
        NQS model that computes log ψ(x).
    hamiltonian:
        Hamiltonian object containing the qubit operator.
    x:
        Batch of configurations, shape (batch, n_qubits).

    Returns
    -------
    E_loc:
        Local energies, shape (batch,).

    Notes
    -----
    This implementation computes the local energy by directly applying the
    Hamiltonian matrix to the wavefunction. For large systems, you would
    use sparse matrix-vector products or operator-specific routines.
    """
    device = x.device
    n_qubits = x.shape[1]
    batch_size = x.shape[0]

    # Convert Hamiltonian to dense matrix (only feasible for small systems)
    ham_matrix = hamiltonian.hamiltonian.to_matrix()  # (2^n_qubits, 2^n_qubits)
    ham_matrix = torch.tensor(ham_matrix, dtype=torch.complex64, device=device)

    # Compute wavefunction for all configurations
    all_configs = enumerate_configurations(n_qubits, device)  # (2^n_qubits, n_qubits)

    with torch.no_grad():
        # Use log_psi method if available (for autoregressive models), otherwise forward
        if hasattr(model, "log_psi"):
            log_psi_all = model.log_psi(all_configs)  # (2^n_qubits,)
        else:
            log_psi_all = model(all_configs)  # (2^n_qubits,)
        psi_all = torch.exp(log_psi_all)  # Real-valued wavefunction

    # Normalize wavefunction
    norm = torch.sqrt(torch.sum(psi_all**2))
    psi_all = psi_all / norm

    # Apply Hamiltonian: H|ψ⟩
    h_psi = ham_matrix @ psi_all.to(torch.complex64)  # (2^n_qubits,)

    # Compute local energy for each input configuration
    # E_loc(x) = ⟨x|H|ψ⟩ / ⟨x|ψ⟩ = (H|ψ⟩)[x] / ψ(x)
    E_loc = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for i in range(batch_size):
        # Find index of configuration x[i] in all_configs
        config_idx = 0
        for j in range(n_qubits):
            if x[i, j] > 0.5:
                config_idx += 2 ** (n_qubits - 1 - j)

        # Local energy
        psi_x = psi_all[config_idx]
        h_psi_x = h_psi[config_idx]

        if abs(psi_x) > 1e-10:
            E_loc[i] = (h_psi_x / psi_x).real
        else:
            E_loc[i] = 0.0

    return E_loc


def compute_energy_exact(
    model: nn.Module,
    hamiltonian: H2Hamiltonian,
    device: torch.device,
) -> float:
    """Compute exact energy expectation value by enumerating all configurations.

    E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩

    Parameters
    ----------
    model:
        NQS model.
    hamiltonian:
        Hamiltonian object.
    device:
        Device for computation.

    Returns
    -------
    energy:
        Energy expectation value.
    """
    n_qubits = hamiltonian.num_qubits
    all_configs = enumerate_configurations(n_qubits, device)

    with torch.no_grad():
        # Use log_psi method if available (for autoregressive models), otherwise forward
        if hasattr(model, "log_psi"):
            log_psi = model.log_psi(all_configs)
        else:
            log_psi = model(all_configs)
        psi = torch.exp(log_psi)
        norm_sq = torch.sum(psi**2)
        psi_normalized = psi / torch.sqrt(norm_sq)

    # Compute ⟨ψ|H|ψ⟩
    ham_matrix = hamiltonian.hamiltonian.to_matrix()
    ham_matrix = torch.tensor(ham_matrix, dtype=torch.complex64, device=device)

    psi_complex = psi_normalized.to(torch.complex64)
    energy = torch.real(psi_complex.conj() @ ham_matrix @ psi_complex)

    return energy.item()


class VMCTrainer:
    """Variational Monte Carlo trainer for NQS models.

    This trainer supports both exact enumeration (for small systems) and
    sampling-based VMC (for larger systems).

    Example
    -------
    >>> from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
    >>> from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    >>>
    >>> # Build model and Hamiltonian
    >>> config = FFNNNQSConfig(n_visible=12, n_hidden=64, n_layers=2)
    >>> model = FFNNNQS(config).to("cuda")
    >>> hamiltonian = build_h2_hamiltonian_12bit(H2Config(bond_length=0.74, bit_depth=12))
    >>>
    >>> # Train
    >>> vmc_config = VMCConfig(learning_rate=1e-3, n_epochs=20, use_exact=True)
    >>> trainer = VMCTrainer(model, hamiltonian, vmc_config)
    >>> history = trainer.train()
    >>>
    >>> print(f"Final energy: {history['energy'][-1]:.6f} Ha")
    >>> print(f"FCI energy: {hamiltonian.fci_energy:.6f} Ha")
    """

    def __init__(
        self,
        model: nn.Module,
        hamiltonian: H2Hamiltonian,
        config: VMCConfig,
    ) -> None:
        """Initialize VMC trainer.

        Parameters
        ----------
        model:
            NQS model to train.
        hamiltonian:
            Molecular Hamiltonian.
        config:
            Training configuration.
        """
        self.model = model
        self.hamiltonian = hamiltonian
        self.config = config

        self.device = next(model.parameters()).device

        # Create optimizer based on config
        if config.optimizer_type.lower() == "sr":
            self.optimizer = SROptimizer(
                model,
                learning_rate=config.learning_rate,
                regularization=config.sr_regularization,
                use_iterative_solver=config.sr_use_iterative,
            )
            self.scheduler = None  # Schedulers not used with SR
        elif config.optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            self.scheduler = self._create_scheduler()
        elif config.optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            self.scheduler = self._create_scheduler()
        else:
            raise ValueError(
                f"Unknown optimizer_type: {config.optimizer_type}. "
                f"Options: 'adam', 'sr', 'sgd'."
            )

        # Pre-enumerate configurations if using exact training
        if config.use_exact:
            if config.use_particle_constraint:
                # Use particle number constraint (粒子數約束)
                n_particles = config.n_particles or hamiltonian.num_electrons
                self.all_configs = enumerate_configurations_fixed_n(
                    hamiltonian.num_qubits, n_particles, self.device
                )
                self.n_particles = n_particles
            else:
                self.all_configs = enumerate_configurations(
                    hamiltonian.num_qubits, self.device
                )
                self.n_particles = None
        else:
            self.all_configs = None
            self.n_particles = None

        self.history: Dict[str, List[float]] = {
            "energy": [],
            "energy_std": [],
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
        }

        # Add SR-specific tracking if using SR
        if config.optimizer_type.lower() == "sr":
            self.history["sr_condition_number"] = []
            self.history["sr_force_norm"] = []
            self.history["sr_update_norm"] = []

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler based on config.

        Returns
        -------
        scheduler:
            Learning rate scheduler or None if scheduler_type is "none".
        """
        if self.config.scheduler_type == "none":
            return None

        elif self.config.scheduler_type == "cosine":
            # Create cosine annealing scheduler with warmup
            def lr_lambda(epoch: int) -> float:
                if epoch < self.config.warmup_epochs:
                    # Linear warmup
                    return (epoch + 1) / self.config.warmup_epochs
                else:
                    # Cosine annealing after warmup
                    progress = (epoch - self.config.warmup_epochs) / max(
                        1, self.config.n_epochs - self.config.warmup_epochs
                    )
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                    # Scale between min_lr and max_lr
                    min_factor = self.config.min_lr / self.config.learning_rate
                    return min_factor + (1 - min_factor) * cosine_decay

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        elif self.config.scheduler_type == "step":
            # Step decay: halve LR every 1/4 of total epochs
            step_size = max(1, self.config.n_epochs // 4)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=0.5
            )

        elif self.config.scheduler_type == "exponential":
            # Exponential decay to reach min_lr at final epoch
            gamma = (self.config.min_lr / self.config.learning_rate) ** (
                1.0 / self.config.n_epochs
            )
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        else:
            raise ValueError(
                f"Unknown scheduler_type: {self.config.scheduler_type}. "
                f"Options: 'none', 'cosine', 'step', 'exponential'."
            )

    def train_step_exact(self) -> Tuple[float, float, float]:
        """Perform one training step using exact enumeration.

        Returns
        -------
        loss:
            Training loss value.
        energy:
            Current energy expectation value.
        grad_norm:
            Gradient norm.
        """
        self.model.train()

        # Enumerate all configurations
        all_configs = self.all_configs
        n_configs = len(all_configs)

        # Compute wavefunction and probabilities for all configurations
        log_psi = self.model(all_configs)  # (n_configs,)
        psi = torch.exp(log_psi)
        norm_sq = torch.sum(psi**2)
        prob = psi**2 / norm_sq  # Normalized probability

        # Compute local energies
        E_loc = compute_local_energy(self.model, self.hamiltonian, all_configs)

        # Compute energy expectation value
        energy = torch.sum(prob * E_loc).item()

        # Compute energy variance (for diagnostics)
        energy_var = torch.sum(prob * (E_loc - energy)**2).item()
        energy_std = np.sqrt(energy_var)

        # Compute loss using correct VMC gradient
        # The gradient for VMC is: ∇E = 2 * Σ_x prob(x) * (E_loc(x) - E) * ∇log ψ(x)
        #
        # IMPORTANT: prob must be detached! If we don't detach prob, the gradient flows
        # through the normalization constant, which creates incorrect gradients that can
        # push energy below the variational bound (violating the variational principle).
        #
        # The correct loss is:
        # loss = 2 * Σ_x prob.detach() * (E_loc(x) - E) * log_psi(x)
        #      = 2 * ⟨(E_loc - E) * log_psi⟩_prob
        #
        # This gives gradient: ∇loss = 2 * Σ_x prob.detach() * (E_loc(x) - E) * ∇log_psi(x)
        loss = 2.0 * torch.sum(prob.detach() * (E_loc - energy) * log_psi)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if self.config.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
        else:
            # Compute gradient norm for logging
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        self.optimizer.step()

        return loss.item(), energy, grad_norm

    def train_step_exact_sr(self) -> Tuple[float, float, float, Dict[str, float]]:
        """Perform one SR training step using exact enumeration.

        Returns
        -------
        loss:
            Training loss value (energy for consistency).
        energy:
            Current energy expectation value.
        grad_norm:
            Parameter update norm (for consistency with Adam).
        sr_stats:
            Dictionary with SR-specific statistics.
        """
        self.model.train()

        # Enumerate all configurations
        all_configs = self.all_configs

        # Compute local energies
        E_loc = compute_local_energy(self.model, self.hamiltonian, all_configs)

        # Compute energy expectation value (for logging)
        with torch.no_grad():
            log_psi = self.model(all_configs)
            psi = torch.exp(log_psi)
            norm_sq = torch.sum(psi**2)
            prob = psi**2 / norm_sq
            energy = torch.sum(prob * E_loc).item()

        # Perform SR step
        # Note: SROptimizer needs gradients enabled for log_psi computation
        sr_stats = self.optimizer.step_with_configs(all_configs, E_loc)

        # Use update_norm as "grad_norm" for consistency with Adam interface
        grad_norm = sr_stats["update_norm"]
        loss = energy  # For logging consistency

        return loss, energy, grad_norm, sr_stats

    def train_step_sampling(self) -> Tuple[float, float, float]:
        """Perform one training step using sampling.

        Returns
        -------
        loss:
            Training loss value.
        energy:
            Current energy estimate.
        grad_norm:
            Gradient norm.
        """
        self.model.train()

        # Sample configurations from the model
        with torch.no_grad():
            samples = self.model.sample(self.config.n_samples, device=self.device)

        # Compute log probabilities and local energies
        log_psi = self.model(samples)
        E_loc = compute_local_energy(self.model, self.hamiltonian, samples)

        # Estimate energy
        energy = torch.mean(E_loc).item()
        energy_std = torch.std(E_loc).item()

        # Compute loss using REINFORCE-style gradient
        # ∇E ≈ (2/N) Σ_i (E_loc(x_i) - E) * ∇log ψ(x_i)
        loss = torch.mean((E_loc - energy) * 2.0 * log_psi)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if self.config.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
        else:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        self.optimizer.step()

        return loss.item(), energy, grad_norm

    def train_step_sampling_sr(self) -> Tuple[float, float, float, float, Dict[str, float]]:
        """Perform one SR training step using sampling.

        Returns
        -------
        loss:
            Training loss value (energy for consistency).
        energy:
            Current energy estimate.
        energy_std:
            Energy standard deviation.
        grad_norm:
            Parameter update norm.
        sr_stats:
            Dictionary with SR-specific statistics.
        """
        self.model.train()

        # Sample configurations from the model
        with torch.no_grad():
            samples = self.model.sample(self.config.n_samples, device=self.device)

        # Compute local energies
        E_loc = compute_local_energy(self.model, self.hamiltonian, samples)

        # Estimate energy
        energy = torch.mean(E_loc).item()
        energy_std = torch.std(E_loc).item()

        # Perform SR step
        sr_stats = self.optimizer.step_with_configs(samples, E_loc)

        # Use update_norm as "grad_norm" for consistency
        grad_norm = sr_stats["update_norm"]
        loss = energy

        return loss, energy, energy_std, grad_norm, sr_stats

    def train(self) -> Dict[str, List[float]]:
        """Run full training loop.

        Returns
        -------
        history:
            Dictionary containing training history with keys:
            - 'energy': **Electronic** energy values per epoch (compare with electronic_fci_energy)
            - 'energy_std': Energy standard deviation per epoch
            - 'loss': Loss values per epoch
            - 'grad_norm': Gradient norms per epoch
            - 'learning_rate': Learning rate per epoch

        Notes
        -----
        The 'energy' values are **electronic** energies (without nuclear repulsion).
        To get total energies, add `hamiltonian.nuclear_repulsion`.
        Compare with `hamiltonian.electronic_fci_energy`, NOT `hamiltonian.fci_energy`.
        """
        # Use the new property for cleaner code
        self.electronic_fci = self.hamiltonian.electronic_fci_energy

        print(f"\nStarting VMC training for {self.config.n_epochs} epochs")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Optimizer: {self.config.optimizer_type.upper()}")
        if self.config.optimizer_type.lower() == "sr":
            print(f"SR regularization: {self.config.sr_regularization}")
        print(f"Hamiltonian: {self.hamiltonian.num_qubits} qubits, {self.hamiltonian.num_electrons} electrons")
        print(f"FCI reference energy: {self.hamiltonian.fci_energy:.6f} Ha (total = electronic + nuclear)")
        print(f"Electronic FCI energy: {self.electronic_fci:.6f} Ha")
        print(f"Nuclear repulsion: {self.hamiltonian.nuclear_repulsion:.6f} Ha")
        print(f"Training mode: {'exact enumeration' if self.config.use_exact else 'sampling'}")
        if self.config.use_particle_constraint:
            print(f"Particle constraint: N={self.n_particles} (configs: {len(self.all_configs)})")
        elif self.config.use_exact:
            print(f"Particle constraint: None (configs: {len(self.all_configs)})")
        if self.config.optimizer_type.lower() != "sr":
            print(f"Scheduler: {self.config.scheduler_type}")
        print(f"Device: {self.device}\n")

        pbar = tqdm(range(self.config.n_epochs), desc="Training")

        for epoch in pbar:
            # Determine which training step to use
            use_sr = self.config.optimizer_type.lower() == "sr"

            if use_sr:
                # SR optimizer
                if self.config.use_exact:
                    loss, energy, grad_norm, sr_stats = self.train_step_exact_sr()
                    energy_std = 0.0
                else:
                    loss, energy, energy_std, grad_norm, sr_stats = self.train_step_sampling_sr()

                # Log SR-specific statistics
                self.history["sr_condition_number"].append(sr_stats["condition_number"])
                self.history["sr_force_norm"].append(sr_stats["force_norm"])
                self.history["sr_update_norm"].append(sr_stats["update_norm"])
                current_lr = self.config.learning_rate  # SR uses fixed learning rate
            else:
                # Standard optimizer (Adam/SGD)
                current_lr = self.optimizer.param_groups[0]["lr"]

                if self.config.use_exact:
                    loss, energy, grad_norm = self.train_step_exact()
                    energy_std = 0.0
                else:
                    loss, energy, grad_norm = self.train_step_sampling()
                    # Recompute energy_std
                    with torch.no_grad():
                        samples = self.model.sample(self.config.n_samples, device=self.device)
                        E_loc = compute_local_energy(self.model, self.hamiltonian, samples)
                        energy_std = torch.std(E_loc).item()

                # Step the scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

            # Log history (common to all optimizers)
            self.history["energy"].append(energy)
            self.history["energy_std"].append(energy_std)
            self.history["loss"].append(loss)
            self.history["grad_norm"].append(grad_norm)
            self.history["learning_rate"].append(current_lr)

            # Update progress bar - compare to electronic FCI
            error = abs(energy - self.electronic_fci)
            postfix = {
                "E": f"{energy:.6f}",
                "Error": f"{error:.6f}",
                "LR": f"{current_lr:.2e}",
                "|∇|": f"{grad_norm:.4f}",
            }
            if use_sr:
                postfix["Cond"] = f"{sr_stats['condition_number']:.1e}"
            pbar.set_postfix(postfix)

        # Final evaluation
        final_energy = compute_energy_exact(self.model, self.hamiltonian, self.device)
        total_energy = final_energy + self.hamiltonian.nuclear_repulsion
        print(f"\nTraining complete!")
        print(f"Final electronic energy: {final_energy:.6f} Ha")
        print(f"Final total energy: {total_energy:.6f} Ha")
        print(f"Electronic FCI: {self.electronic_fci:.6f} Ha")
        print(f"Total FCI: {self.hamiltonian.fci_energy:.6f} Ha")
        print(f"Error vs electronic FCI: {abs(final_energy - self.electronic_fci):.6f} Ha")

        return self.history


class TransformerVMCTrainer(VMCTrainer):
    """VMC trainer specialized for autoregressive Transformer NQS models.

    This trainer handles the autoregressive probability factorization correctly:
        p(σ) = ∏_i p(σ_i | σ_{<i})
        ψ(σ) ∝ √p(σ)
        log ψ(σ) = (1/2) * Σ_i log p(σ_i | σ_{<i})

    The gradient for VMC is:
        ∇E = 2 * ⟨(E_loc - ⟨E⟩) * ∇log ψ⟩

    For autoregressive models, ∇log ψ = (1/2) * ∇log p, where log p is computed
    via the teacher-forced cross-entropy.

    Example
    -------
    >>> from src.nqs_models.transformer_nqs import TransformerNQS, TransformerNQSConfig
    >>> from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    >>>
    >>> # Build model and Hamiltonian
    >>> config = TransformerNQSConfig(n_visible=12, d_model=128, n_heads=4, n_layers=2)
    >>> model = TransformerNQS(config).to("cuda")
    >>> hamiltonian = build_h2_hamiltonian_12bit(H2Config(bond_length=0.74, bit_depth=12))
    >>>
    >>> # Train with autoregressive VMC
    >>> vmc_config = VMCConfig(learning_rate=1e-3, n_epochs=20, use_exact=True)
    >>> trainer = TransformerVMCTrainer(model, hamiltonian, vmc_config)
    >>> history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        hamiltonian: H2Hamiltonian,
        config: VMCConfig,
    ) -> None:
        """Initialize Transformer VMC trainer.

        Parameters
        ----------
        model:
            Autoregressive Transformer NQS model to train.
        hamiltonian:
            Molecular Hamiltonian.
        config:
            Training configuration.
        """
        super().__init__(model, hamiltonian, config)

        # Verify model has the required method
        if not hasattr(model, "log_psi"):
            raise ValueError(
                "Model must have a 'log_psi' method for autoregressive VMC training. "
                "Use TransformerNQS with the updated implementation."
            )

    def train_step_exact(self) -> Tuple[float, float, float]:
        """Perform one training step using exact enumeration with autoregressive model.

        This implements the proper VMC gradient for autoregressive models:
            ∇E = 2 * Σ_x prob(x) * (E_loc(x) - E) * ∇log ψ(x)

        where log ψ(x) = (1/2) * Σ_i log p(σ_i | σ_{<i})

        Returns
        -------
        loss:
            Training loss value.
        energy:
            Current energy expectation value.
        grad_norm:
            Gradient norm.
        """
        self.model.train()

        # Enumerate all configurations
        all_configs = self.all_configs
        n_configs = len(all_configs)

        # Compute autoregressive log ψ for all configurations
        log_psi = self.model.log_psi(all_configs)  # (n_configs,)
        psi = torch.exp(log_psi)

        # Compute normalized probabilities
        norm_sq = torch.sum(psi**2)
        prob = psi**2 / norm_sq  # Normalized probability

        # Compute local energies
        E_loc = compute_local_energy(self.model, self.hamiltonian, all_configs)

        # Compute energy expectation value
        energy = torch.sum(prob * E_loc).item()

        # Compute energy variance (for diagnostics)
        energy_var = torch.sum(prob * (E_loc - energy)**2).item()
        energy_std = np.sqrt(energy_var)

        # Compute loss using correct VMC gradient
        # The gradient for VMC is: ∇E = 2 * Σ_x prob(x) * (E_loc(x) - E) * ∇log ψ(x)
        #
        # IMPORTANT: prob must be detached! If we don't detach prob, the gradient flows
        # through the normalization constant, which creates incorrect gradients that can
        # push energy below the variational bound (violating the variational principle).
        loss = 2.0 * torch.sum(prob.detach() * (E_loc - energy) * log_psi)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if self.config.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
        else:
            # Compute gradient norm for logging
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        self.optimizer.step()

        return loss.item(), energy, grad_norm

    def train_step_sampling(self) -> Tuple[float, float, float]:
        """Perform one training step using sampling with autoregressive model.

        Returns
        -------
        loss:
            Training loss value.
        energy:
            Current energy estimate.
        grad_norm:
            Gradient norm.
        """
        self.model.train()

        # Sample configurations from the model autoregressively
        with torch.no_grad():
            samples = self.model.sample(self.config.n_samples, device=self.device)

        # Compute log ψ and local energies
        log_psi = self.model.log_psi(samples)  # (n_samples,)
        E_loc = compute_local_energy(self.model, self.hamiltonian, samples)

        # Estimate energy
        energy = torch.mean(E_loc).item()
        energy_std = torch.std(E_loc).item()

        # Compute REINFORCE-style gradient for VMC
        # ∇E ≈ (2/N) Σ_i (E_loc(x_i) - E) * ∇log ψ(x_i)
        # We use the centered gradient estimator to reduce variance
        loss = torch.mean((E_loc.detach() - energy) * 2.0 * log_psi)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if self.config.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
        else:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        self.optimizer.step()

        return loss.item(), energy, grad_norm


class ComplexVMCTrainer(VMCTrainer):
    """VMC trainer for complex-valued NQS models.

    This trainer handles complex wavefunctions ψ(x) = |ψ(x)| * exp(i*φ(x)) where
    both amplitude and phase are learned. This solves the "sign problem" that
    limits real-valued NQS.

    The VMC gradient for complex wavefunctions is:
        ∇E = 2 * Re[ Σ_x |ψ(x)|² (E_loc(x) - E)^* ∇log ψ(x) ]

    where ∇log ψ(x) = ∇log|ψ(x)| + i * ∇φ(x)

    For H₂ ground state: |ψ_GS⟩ ≈ 0.99|1100⟩ - 0.14|0011⟩
    The negative sign is represented by φ(0011) = π while φ(1100) = 0.

    Example
    -------
    >>> from src.nqs_models.ffn_nqs import ComplexFFNNNQS, FFNNNQSConfig
    >>> from src.sqd_interface.hamiltonian import build_h2_hamiltonian_12bit, H2Config
    >>>
    >>> # Build model and Hamiltonian
    >>> config = FFNNNQSConfig(n_visible=4, n_hidden=64, n_layers=2)
    >>> model = ComplexFFNNNQS(config).to("cuda")
    >>> hamiltonian = build_h2_hamiltonian_12bit(H2Config(bond_length=0.74, bit_depth=4))
    >>>
    >>> # Train with complex VMC
    >>> vmc_config = VMCConfig(learning_rate=1e-3, n_epochs=100, use_exact=True)
    >>> trainer = ComplexVMCTrainer(model, hamiltonian, vmc_config)
    >>> history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        hamiltonian: H2Hamiltonian,
        config: VMCConfig,
    ) -> None:
        """Initialize Complex VMC trainer.

        Parameters
        ----------
        model:
            Complex NQS model (e.g., ComplexFFNNNQS) with forward() returning
            (log_amp, phase).
        hamiltonian:
            Molecular Hamiltonian.
        config:
            Training configuration.
        """
        super().__init__(model, hamiltonian, config)

        # Verify model has the expected interface
        test_input = torch.zeros(1, hamiltonian.num_qubits, device=self.device)
        output = model(test_input)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError(
                "Complex NQS model must return (log_amp, phase) tuple from forward(). "
                "Got: {}".format(type(output))
            )

    def train_step_exact(self) -> Tuple[float, float, float]:
        """Perform one training step using exact enumeration for complex NQS.

        For complex wavefunctions, the VMC gradient is:
            ∇E = 2 * Re[ Σ_x prob(x) * (E_loc(x) - E)^* * ∇log ψ(x) ]

        where:
            - prob(x) = |ψ(x)|² / Σ|ψ|²
            - ∇log ψ(x) = ∇log|ψ(x)| + i * ∇φ(x)
            - E_loc(x) is complex in general

        Returns
        -------
        loss:
            Training loss value.
        energy:
            Current energy expectation value.
        grad_norm:
            Gradient norm.
        """
        self.model.train()

        all_configs = self.all_configs

        # Compute log-amplitude and phase for all configurations
        log_amp, phase = self.model(all_configs)  # Both (n_configs,)

        # Compute probability from amplitude only
        # prob(x) = |ψ(x)|² / Z = exp(2*log_amp) / Σ exp(2*log_amp)
        log_prob_unnorm = 2.0 * log_amp
        log_prob_norm = log_prob_unnorm - torch.logsumexp(log_prob_unnorm, dim=0)
        prob = torch.exp(log_prob_norm)

        # Compute complex local energies
        E_loc = compute_local_energy_complex(self.model, self.hamiltonian, all_configs)

        # Compute energy expectation (should be real for Hermitian H)
        energy_complex = torch.sum(prob * E_loc)
        energy = energy_complex.real.item()

        # VMC gradient for complex wavefunctions:
        # ∇E = 2 * Re[ Σ_x prob(x) * (E_loc(x) - E)^* * ∇log ψ(x) ]
        #
        # Since log ψ(x) = log_amp(x) + i * phase(x), we have:
        # ∇E = 2 * Re[ Σ_x prob(x) * (E_loc(x) - E)^* * (∇log_amp + i*∇phase) ]
        #    = 2 * Σ_x prob(x) * [ Re(E_loc - E) * ∇log_amp + Im(E_loc - E) * ∇phase ]
        #
        # Simplifying for gradient descent:
        # loss_amp = 2 * Σ_x prob * Re(E_loc - E) * log_amp
        # loss_phase = 2 * Σ_x prob * Im(E_loc - E) * phase

        E_loc_centered = E_loc - energy
        E_loc_re = E_loc_centered.real
        E_loc_im = E_loc_centered.imag

        # Loss that gives correct gradients when backpropagated
        # Note: prob must be detached to avoid gradient through normalization
        loss_amp = 2.0 * torch.sum(prob.detach() * E_loc_re * log_amp)
        loss_phase = 2.0 * torch.sum(prob.detach() * E_loc_im * phase)
        loss = loss_amp + loss_phase

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if self.config.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
        else:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        self.optimizer.step()

        return loss.item(), energy, grad_norm

    def train(self) -> Dict[str, List[float]]:
        """Run full training loop for complex NQS.

        Returns
        -------
        history:
            Dictionary containing training history.
        """
        self.electronic_fci = self.hamiltonian.electronic_fci_energy

        print(f"\nStarting Complex VMC training for {self.config.n_epochs} epochs")
        print(f"Model: {self.model.__class__.__name__} (complex wavefunction)")
        print(f"Optimizer: {self.config.optimizer_type.upper()}")
        print(f"Hamiltonian: {self.hamiltonian.num_qubits} qubits, {self.hamiltonian.num_electrons} electrons")
        print(f"Electronic FCI energy: {self.electronic_fci:.6f} Ha")
        print(f"Training mode: {'exact enumeration' if self.config.use_exact else 'sampling'}")
        if self.config.use_particle_constraint:
            print(f"Particle constraint: N={self.n_particles} (configs: {len(self.all_configs)})")
        elif self.config.use_exact:
            print(f"Particle constraint: None (configs: {len(self.all_configs)})")
        if self.config.optimizer_type.lower() != "sr":
            print(f"Scheduler: {self.config.scheduler_type}")
        print(f"Device: {self.device}\n")

        pbar = tqdm(range(self.config.n_epochs), desc="Training (Complex)")

        for epoch in pbar:
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.config.use_exact:
                loss, energy, grad_norm = self.train_step_exact()
                energy_std = 0.0
            else:
                # Sampling-based not implemented for complex yet
                raise NotImplementedError(
                    "Sampling-based training not yet implemented for complex NQS. "
                    "Use use_exact=True for now."
                )

            # Step the scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log history
            self.history["energy"].append(energy)
            self.history["energy_std"].append(energy_std)
            self.history["loss"].append(loss)
            self.history["grad_norm"].append(grad_norm)
            self.history["learning_rate"].append(current_lr)

            # Update progress bar
            error = abs(energy - self.electronic_fci)
            pbar.set_postfix({
                "E": f"{energy:.6f}",
                "Error": f"{error:.6f}",
                "LR": f"{current_lr:.2e}",
                "|∇|": f"{grad_norm:.4f}",
            })

        # Final evaluation using complex energy computation
        final_energy = compute_energy_exact_complex(self.model, self.hamiltonian, self.device)
        total_energy = final_energy + self.hamiltonian.nuclear_repulsion
        print(f"\nTraining complete!")
        print(f"Final electronic energy: {final_energy:.6f} Ha")
        print(f"Final total energy: {total_energy:.6f} Ha")
        print(f"Electronic FCI: {self.electronic_fci:.6f} Ha")
        print(f"Error vs electronic FCI: {abs(final_energy - self.electronic_fci):.6f} Ha")

        return self.history
