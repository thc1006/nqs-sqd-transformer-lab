"""Stochastic Reconfiguration (SR) optimizer for Neural Quantum States.

SR is a natural gradient method that uses the quantum Fisher information matrix
to guide parameter updates. It is particularly effective for variational quantum
state optimization.

Theory
------
For a parameterized quantum state |ψ(θ)⟩, Stochastic Reconfiguration computes:

1. Log-derivatives (force operators):
   O_k(x) = ∂ log ψ(x) / ∂θ_k

2. Quantum Fisher Information Matrix (metric tensor):
   S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩
   where ⟨·⟩ denotes expectation over |ψ|²

3. Energy gradient (force vector):
   F_k = -2 * ⟨(E_loc - ⟨E⟩) * O_k⟩

4. Natural gradient update:
   (S + ε*I) * δθ = F
   θ_new = θ + η * δθ

The regularization ε*I stabilizes the matrix inversion when S is ill-conditioned.

References
----------
- S. Sorella, Phys. Rev. Lett. 80, 4558 (1998)
  "Generalized Lanczos algorithm for variational quantum Monte Carlo"
- G. Carleo & M. Troyer, Science 355, 602 (2017)
  "Solving the quantum many-body problem with artificial neural networks"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class SROptimizer:
    """Stochastic Reconfiguration optimizer for NQS variational optimization.

    This optimizer implements the natural gradient descent using the quantum
    Fisher information matrix. It is more stable and efficient than standard
    gradient descent for variational quantum states.

    Parameters
    ----------
    model:
        Neural network model (NQS) to optimize.
    learning_rate:
        Step size for parameter updates (η).
    regularization:
        Regularization parameter (ε) for stabilizing matrix inversion.
    use_iterative_solver:
        If True, use conjugate gradient instead of direct inversion.
        Recommended for large models (>10k parameters).
    cg_maxiter:
        Maximum iterations for conjugate gradient solver.
    cg_tol:
        Convergence tolerance for conjugate gradient.

    Example
    -------
    >>> model = FFNNNQS(config).to("cuda")
    >>> optimizer = SROptimizer(model, learning_rate=0.05, regularization=0.01)
    >>>
    >>> # In training loop:
    >>> log_psi, log_derivs = compute_log_derivs(model, configs)
    >>> energy_local = compute_local_energies(model, hamiltonian, configs)
    >>> optimizer.step(log_derivs, energy_local)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        use_iterative_solver: bool = False,
        cg_maxiter: int = 100,
        cg_tol: float = 1e-5,
    ) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.use_iterative_solver = use_iterative_solver
        self.cg_maxiter = cg_maxiter
        self.cg_tol = cg_tol

        # Count total number of parameters
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get device from model
        self.device = next(model.parameters()).device

        # Statistics for monitoring
        self.stats: Dict[str, List[float]] = {
            "condition_number": [],
            "force_norm": [],
            "update_norm": [],
            "solver_iterations": [],
        }

    def zero_grad(self) -> None:
        """Clear gradients (for compatibility with standard optimizer interface)."""
        self.model.zero_grad()

    def compute_log_derivatives(
        self,
        configs: torch.Tensor,
        log_psi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log-derivatives O_k(x) = ∂ log ψ(x) / ∂θ_k for each configuration.

        Parameters
        ----------
        configs:
            Batch of configurations, shape (n_samples, n_visible).
        log_psi:
            Pre-computed log ψ values, shape (n_samples,).
            If None, will compute from model.

        Returns
        -------
        log_derivs:
            Log-derivatives for each sample and parameter.
            Shape: (n_samples, n_params)
        """
        n_samples = configs.shape[0]

        # Compute log ψ if not provided
        if log_psi is None:
            log_psi = self.model(configs)

        # Compute gradients for each sample
        log_derivs = torch.zeros(
            n_samples, self.n_params, dtype=torch.float32, device=self.device
        )

        for i in range(n_samples):
            # Clear previous gradients
            self.model.zero_grad()

            # Compute gradient of log ψ(x_i) w.r.t. parameters
            log_psi[i].backward(retain_graph=True)

            # Collect gradients into flat vector
            param_idx = 0
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    n_elem = param.numel()
                    log_derivs[i, param_idx:param_idx + n_elem] = param.grad.view(-1)
                    param_idx += n_elem

        # Clear gradients after collection
        self.model.zero_grad()

        return log_derivs

    def compute_fisher_matrix(
        self, log_derivs: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantum Fisher information matrix S_ij.

        S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩

        Parameters
        ----------
        log_derivs:
            Log-derivatives for each sample, shape (n_samples, n_params).

        Returns
        -------
        S:
            Fisher matrix, shape (n_params, n_params).
        """
        n_samples = log_derivs.shape[0]

        # Compute centered log-derivatives: O_k - ⟨O_k⟩
        O_mean = torch.mean(log_derivs, dim=0, keepdim=True)  # (1, n_params)
        O_centered = log_derivs - O_mean  # (n_samples, n_params)

        # S = (1/N) Σ_x (O - ⟨O⟩) ⊗ (O - ⟨O⟩)
        # This is equivalent to: S = Cov(O)
        S = (O_centered.T @ O_centered) / n_samples  # (n_params, n_params)

        return S

    def compute_force_vector(
        self,
        log_derivs: torch.Tensor,
        energy_local: torch.Tensor,
    ) -> torch.Tensor:
        """Compute force vector F_k = -2 * ⟨(E_loc - ⟨E⟩) * O_k⟩.

        Parameters
        ----------
        log_derivs:
            Log-derivatives for each sample, shape (n_samples, n_params).
        energy_local:
            Local energies for each sample, shape (n_samples,).

        Returns
        -------
        F:
            Force vector, shape (n_params,).
        """
        # Compute energy expectation
        E_mean = torch.mean(energy_local)

        # Compute centered local energies
        E_centered = energy_local - E_mean  # (n_samples,)

        # F_k = -2 * ⟨(E_loc - ⟨E⟩) * O_k⟩
        # = -2 * (1/N) Σ_x (E_loc(x) - ⟨E⟩) * O_k(x)
        F = -2.0 * torch.mean(
            E_centered.unsqueeze(-1) * log_derivs, dim=0
        )  # (n_params,)

        return F

    def solve_linear_system(
        self, S: torch.Tensor, F: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Solve (S + ε*I) * δθ = F for parameter update δθ.

        Parameters
        ----------
        S:
            Fisher matrix, shape (n_params, n_params).
        F:
            Force vector, shape (n_params,).

        Returns
        -------
        delta_theta:
            Parameter update, shape (n_params,).
        n_iter:
            Number of iterations (for iterative solver) or -1 for direct.
        """
        # Add regularization: S_reg = S + ε*I
        S_reg = S + self.regularization * torch.eye(
            self.n_params, dtype=S.dtype, device=S.device
        )

        if self.use_iterative_solver:
            # Use conjugate gradient for large systems
            delta_theta, n_iter = self._solve_cg(S_reg, F)
        else:
            # Direct solve using Cholesky decomposition (more stable than LU)
            try:
                # Try Cholesky first (assumes S is positive definite)
                L = torch.linalg.cholesky(S_reg)
                delta_theta = torch.cholesky_solve(F.unsqueeze(-1), L).squeeze(-1)
                n_iter = -1
            except RuntimeError:
                # Fall back to LU decomposition if Cholesky fails
                delta_theta = torch.linalg.solve(S_reg, F)
                n_iter = -1

        return delta_theta, n_iter

    def _solve_cg(
        self, A: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Solve Ax = b using conjugate gradient method.

        Parameters
        ----------
        A:
            Symmetric positive-definite matrix, shape (n, n).
        b:
            Right-hand side, shape (n,).

        Returns
        -------
        x:
            Solution, shape (n,).
        n_iter:
            Number of iterations performed.
        """
        x = torch.zeros_like(b)
        r = b - A @ x
        p = r.clone()
        rsold = r @ r

        for i in range(self.cg_maxiter):
            Ap = A @ p
            alpha = rsold / (p @ Ap + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r @ r

            if torch.sqrt(rsnew) < self.cg_tol:
                return x, i + 1

            beta = rsnew / (rsold + 1e-10)
            p = r + beta * p
            rsold = rsnew

        return x, self.cg_maxiter

    def step(
        self,
        log_derivs: torch.Tensor,
        energy_local: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one SR optimization step.

        This method:
        1. Computes Fisher matrix S from log-derivatives
        2. Computes force vector F from energy and log-derivatives
        3. Solves (S + ε*I) * δθ = F
        4. Updates parameters: θ ← θ + η * δθ

        Parameters
        ----------
        log_derivs:
            Log-derivatives O_k(x) for each sample, shape (n_samples, n_params).
        energy_local:
            Local energies E_loc(x) for each sample, shape (n_samples,).

        Returns
        -------
        step_info:
            Dictionary containing:
            - 'condition_number': Condition number of S
            - 'force_norm': Norm of force vector
            - 'update_norm': Norm of parameter update
            - 'n_iter': Number of solver iterations (if applicable)
        """
        # 1. Compute Fisher matrix
        S = self.compute_fisher_matrix(log_derivs)

        # 2. Compute force vector
        F = self.compute_force_vector(log_derivs, energy_local)

        # 3. Solve for parameter update
        delta_theta, n_iter = self.solve_linear_system(S, F)

        # 4. Apply parameter update: θ ← θ + η * δθ
        param_idx = 0
        for param in self.model.parameters():
            if param.requires_grad:
                n_elem = param.numel()
                update = delta_theta[param_idx:param_idx + n_elem].view(param.shape)
                param.data.add_(update, alpha=self.learning_rate)
                param_idx += n_elem

        # Compute diagnostics
        with torch.no_grad():
            # Condition number of S (before regularization)
            try:
                S_eigs = torch.linalg.eigvalsh(S)
                cond = (S_eigs.max() / (S_eigs.min() + 1e-10)).item()
            except:
                cond = -1.0

            force_norm = torch.norm(F).item()
            update_norm = torch.norm(delta_theta).item()

        # Store statistics
        self.stats["condition_number"].append(cond)
        self.stats["force_norm"].append(force_norm)
        self.stats["update_norm"].append(update_norm)
        self.stats["solver_iterations"].append(n_iter)

        return {
            "condition_number": cond,
            "force_norm": force_norm,
            "update_norm": update_norm,
            "n_iter": n_iter,
        }

    def step_with_configs(
        self,
        configs: torch.Tensor,
        energy_local: torch.Tensor,
    ) -> Dict[str, float]:
        """Convenience method that computes log-derivatives and performs SR step.

        Parameters
        ----------
        configs:
            Batch of configurations, shape (n_samples, n_visible).
        energy_local:
            Local energies for each sample, shape (n_samples,).

        Returns
        -------
        step_info:
            Dictionary with step diagnostics.
        """
        # Compute log ψ and log-derivatives
        log_psi = self.model(configs)
        log_derivs = self.compute_log_derivatives(configs, log_psi)

        # Perform SR step
        return self.step(log_derivs, energy_local)

    def get_stats(self) -> Dict[str, List[float]]:
        """Get optimization statistics history.

        Returns
        -------
        stats:
            Dictionary containing lists of:
            - 'condition_number': Condition numbers of Fisher matrix
            - 'force_norm': Norms of force vectors
            - 'update_norm': Norms of parameter updates
            - 'solver_iterations': Solver iterations (if applicable)
        """
        return self.stats.copy()
