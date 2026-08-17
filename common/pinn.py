"""Physics-informed neural network for the reaction-diffusion PDE."""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from .config import Params


class PINN(nn.Module):
    """MLP u(x, t) approximating the PDE solution.

    The residual skip (added after each odd Tanh activation) is kept exactly as in the
    original experiments for reproducibility.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 5, use_residual: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, t], dim=1)
        for i, layer in enumerate(self.net[:-1]):
            out = layer(h)
            if (
                self.use_residual
                and isinstance(layer, nn.Tanh)
                and i > 0
                and i % 2 == 1
                and out.shape[-1] == h.shape[-1]
            ):
                out = out + h
            h = out
        return self.net[-1](h)


def physics_loss(model: PINN, x: torch.Tensor, t: torch.Tensor, params: Params) -> torch.Tensor:
    """Mean squared PDE residual f = u_t - D*u_xx - (p + q*u)(1 - u)."""
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones, create_graph=True)[0]
    residual = u_t - params.D * u_xx - (params.p + params.q * u) * (1.0 - u)
    return torch.mean(residual**2)


def train_pinn(
    model: PINN,
    x_data: torch.Tensor,
    t_data: torch.Tensor,
    u_train: torch.Tensor,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    params: Params,
    epochs: int = 2000,
    lr: float = 1e-3,
    step_size: int = 500,
    gamma: float = 0.9,
    clip: float = 1.0,
    lambda_initial: float = 0.1,
    lambda_final: float = 2.0,
    lambda_warmup: int = 500,
    print_every: int = 500,
) -> tuple[list[tuple[float, float, float, float]], float]:
    """Train the PINN with data loss + physics loss and a linear lambda warmup schedule."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    history: list[tuple[float, float, float, float]] = []
    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        lam = (
            lambda_initial + (lambda_final - lambda_initial) * (epoch / lambda_warmup)
            if epoch < lambda_warmup
            else lambda_final
        )
        loss_data = torch.mean((model(x_data, t_data) - u_train) ** 2)
        loss_phy = physics_loss(model, x_col, t_col, params)
        loss = loss_data + lam * loss_phy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        scheduler.step()
        history.append((loss.item(), loss_data.item(), loss_phy.item(), lam))
        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f} "
                f"(Data: {loss_data.item():.6f}, Phy: {loss_phy.item():.6f}, "
                f"lambda={lam:.3f}, LR={optimizer.param_groups[0]['lr']:.2e})"
            )
    return history, time.time() - start
