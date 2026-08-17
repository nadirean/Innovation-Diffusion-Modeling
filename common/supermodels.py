"""Ensemble modeling: SuperModel ODE and SuperNet PINN (task 7)."""
from __future__ import annotations

import time

import numpy as np
import torch
from numpy.typing import NDArray

from .config import Params
from .pinn import PINN, physics_loss


def make_supermodel_params(base: Params, perturbations: list[tuple[float, float]]) -> list[Params]:
    """Perturbed (p, q) members around the base parameters."""
    return [Params(D=base.D, p=base.p * fp, q=base.q * fq) for fp, fq in perturbations]


def supermodel_ode_system(
    t: float,
    y: NDArray[np.float64],
    members: list[Params],
    coupling: float,
    proc_noise: NDArray[np.float64],
    t_grid: NDArray[np.float64],
) -> list[float]:
    """RHS of the coupled ODE ensemble: dU_i/dt = f(U_i; p_i, q_i) + C*(U_mean - U_i) + noise_i(t)."""
    mean = float(np.mean(y))
    derivatives = []
    for i, params in enumerate(members):
        f = (params.p + params.q * y[i]) * (1.0 - y[i])
        derivatives.append(f + coupling * (mean - y[i]) + np.interp(t, t_grid, proc_noise[i]))
    return derivatives


def train_supernet(
    models: list[PINN],
    x_data: torch.Tensor,
    t_data: torch.Tensor,
    u_train: torch.Tensor,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    members: list[Params],
    base_D: float,
    epochs: int = 2000,
    lr: float = 1e-3,
    step_size: int = 500,
    gamma: float = 0.9,
    clip: float = 1.0,
    lambda_couple_final: float = 0.5,
    warmup_epochs: int = 500,
    print_every: int = 500,
) -> tuple[list[tuple[float, float]], float]:
    """Train an ensemble of PINNs with data loss, per-member physics loss and coupling loss.

    The coupling weight warms up linearly from 0.2*lambda_couple_final to lambda_couple_final.
    """
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=step_size, gamma=gamma) for o in optimizers]
    lambda_couple_init = 0.2 * lambda_couple_final
    history: list[tuple[float, float]] = []
    start = time.time()

    for epoch in range(epochs):
        for opt in optimizers:
            opt.zero_grad()
        lam = (
            lambda_couple_init + (lambda_couple_final - lambda_couple_init) * (epoch / warmup_epochs)
            if epoch < warmup_epochs
            else lambda_couple_final
        )

        u_preds_data = [m(x_data, t_data) for m in models]
        losses_data = [torch.mean((up - u_train) ** 2) for up in u_preds_data]

        u_preds_col = [m(x_col, t_col) for m in models]
        losses_phy = [physics_loss(m, x_col, t_col, member) for m, member in zip(models, members)]

        stack = torch.stack(u_preds_col, dim=0)
        loss_couple = torch.mean((stack - torch.mean(stack, dim=0)) ** 2)

        total_loss = sum(losses_data) + sum(losses_phy) + len(models) * lam * loss_couple
        total_loss.backward()
        for m in models:
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=clip)
        for opt, scheduler in zip(optimizers, schedulers):
            opt.step()
            scheduler.step()

        history.append((total_loss.item(), lam))
        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss.item():.6f}, "
                f"lambda_couple={lam:.3f}, LR={optimizers[0].param_groups[0]['lr']:.2e}"
            )

    return history, time.time() - start
