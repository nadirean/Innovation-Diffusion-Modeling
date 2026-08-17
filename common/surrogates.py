"""ODE surrogate and neural-network surrogates for the spatial mean U(t)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, TensorDataset


def solve_surrogate_ode(
    p: float,
    q: float,
    u0: float,
    t_span: tuple[float, float],
    t_eval: NDArray[np.float64] | None = None,
) -> object:
    """Solve dU/dt = (p + qU)(1 - U); dense output when t_eval is None."""
    return solve_ivp(
        lambda t, U: (p + q * U) * (1.0 - U),
        t_span,
        [u0],
        t_eval=t_eval,
        dense_output=(t_eval is None),
    )


class SurrogateNN(nn.Module):
    """MLP surrogate U(t) for a single parameter set (input: normalized time)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ParameterizedSurrogateNN(nn.Module):
    """MLP surrogate (t, D, p, q) -> U(t) trained across many parameter sets."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Standardizer:
    """Z-score normalization with fit/apply/invert."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    @classmethod
    def fit(cls, x: NDArray[np.float64], eps: float = 1e-8) -> "Standardizer":
        return cls(np.mean(x, axis=0), np.std(x, axis=0) + eps)

    def apply(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x - self.mean) / self.std

    def invert(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return y * self.std + self.mean


def train_surrogate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 2000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    batch_size: int | None = None,
    log_every: int = 200,
) -> None:
    """Train a surrogate network with MSE + Adam (model is mutated in place)."""
    if batch_size is None:
        batches = [(x, y)]
    else:
        batches = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    loss: torch.Tensor | None = None
    for epoch in range(epochs):
        for xb, yb in batches:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % log_every == 0 and loss is not None:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")
