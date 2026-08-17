"""Synthetic data generation for surrogates and PINN training."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import Params
from .model import pde_mean_series


def add_noise(u: NDArray[np.float64], kind: str, level: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Add measurement noise with amplitude level * max(1e-3, |u|).

    kinds: 'gaussian', 'laplace', 'gaussian_mixture'.
    """
    scale = level * np.maximum(1e-3, np.abs(u))
    if kind == "gaussian":
        noise = rng.normal(0.0, scale)
    elif kind == "laplace":
        noise = rng.laplace(0.0, scale)
    elif kind == "gaussian_mixture":
        comp = rng.choice([0, 1, 2], size=u.shape, p=[0.7, 0.2, 0.1])
        noise = np.where(
            comp == 0,
            rng.normal(0.0, scale),
            np.where(comp == 1, rng.laplace(0.0, 1.5 * scale), rng.normal(0.0, 3.0 * scale)),
        )
    else:
        raise ValueError(f"unknown noise kind: {kind}")
    return u + noise


def sample_measurements(
    x: NDArray[np.float64],
    t: NDArray[np.float64],
    u_noisy: NDArray[np.float64],
    n_data: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Random measurement points: (X, u) with X columns (x, t); u_noisy is (nx, nt)."""
    x_grid, t_grid = np.meshgrid(x, t)
    x_flat = x_grid.flatten()[:, None]
    t_flat = t_grid.flatten()[:, None]
    u_flat = u_noisy.T.flatten()[:, None]
    idx = rng.choice(x_flat.shape[0], n_data, replace=False)
    return np.hstack((x_flat[idx], t_flat[idx])), u_flat[idx]


def sample_collocation(n_col: int, tend: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Uniform random collocation points with columns (x, t)."""
    return np.hstack((rng.uniform(0, 1, (n_col, 1)), rng.uniform(0, tend, (n_col, 1))))


def generate_parameterized_dataset(
    num_simulations: int,
    tend: float,
    dt: float,
    nx: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(t, D, p, q) -> U(t) samples from PDE runs with uniformly random parameters."""
    rng = np.random.default_rng(seed)
    inputs: list[list[float]] = []
    outputs: list[list[float]] = []
    for _ in range(num_simulations):
        params = Params(D=rng.uniform(0.01, 1.0), p=rng.uniform(0.001, 0.1), q=rng.uniform(0.1, 10.0))
        t_series, means = pde_mean_series(params, tend, dt, nx)
        for t_val, mean in zip(t_series, means):
            inputs.append([t_val, params.D, params.p, params.q])
            outputs.append([mean])
    return np.asarray(inputs), np.asarray(outputs)
