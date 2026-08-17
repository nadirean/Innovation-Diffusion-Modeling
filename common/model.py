"""Core model: reaction term, Neumann Laplacian, initial conditions, PDE simulation."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from .config import Params


def reaction(u: NDArray[np.float64], p: float, q: float) -> NDArray[np.float64]:
    """Reaction term R(u) = (p + q*u)*(1-u)."""
    return (p + q * u) * (1 - u)


def neumann_laplacian(nx: int, dx: float) -> sp.csr_matrix:
    """Second-derivative operator with homogeneous Neumann BCs on [0, 1]."""
    main = -2.0 * np.ones(nx)
    off = 1.0 * np.ones(nx - 1)
    a = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    a[0, 0] = -1.0
    a[-1, -1] = -1.0
    return a / (dx * dx)


def initial_condition(nx: int, kind: str = "gaussian") -> NDArray[np.float64]:
    """Initial profile u(x, 0) on a grid of nx points.

    kinds: 'gaussian', 'localized', 'small_random', 'narrow_gaussian'.
    """
    x = np.linspace(0, 1, nx)
    if kind == "gaussian":
        u0 = 0.01 * np.ones_like(x)
        u0 += 0.5 * np.exp(-((x - 0.5) ** 2) / (2 * 0.02**2))
        return u0
    if kind == "localized":
        u0 = np.zeros_like(x)
        u0[np.abs(x - 0.5) < 0.05] = 0.5
        return u0 + 1e-3
    if kind == "small_random":
        rng = np.random.default_rng(0)
        return 1e-3 + 0.01 * rng.random(nx)
    if kind == "narrow_gaussian":
        return np.exp(-100.0 * (x - 0.5) ** 2)
    raise ValueError(f"unknown init kind: {kind}")


def pde_snapshots(
    params: Params,
    tend: float,
    nx: int,
    dt: float,
    init_kind: str = "localized",
    save_every: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Crank-Nicolson solve (implicit diffusion, explicit reaction) with Neumann BCs.

    Returns (x, t, U) with U of shape (nx, nt); snapshots taken every save_every steps
    plus the initial state.
    """
    dx = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx)
    u = initial_condition(nx, init_kind)

    a = neumann_laplacian(nx, dx)
    m_l = sp.eye(nx) - 0.5 * dt * params.D * a
    m_r = sp.eye(nx) + 0.5 * dt * params.D * a
    solve = spla.splu(m_l.tocsc()).solve

    states = [u.copy()]
    times = [0.0]
    t = 0.0
    for n in range(int(np.ceil(tend / dt))):
        u = solve(m_r.dot(u) + dt * reaction(u, params.p, params.q))
        t += dt
        if (n + 1) % save_every == 0:
            states.append(u.copy())
            times.append(t)

    return x, np.asarray(times), np.asarray(states).T


def pde_mean_series(
    params: Params,
    tend: float,
    dt: float,
    nx: int,
    init_kind: str = "gaussian",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Spatial mean U(t) at every step of a Crank-Nicolson run (surrogate training data)."""
    dx = 1.0 / (nx - 1)
    a = neumann_laplacian(nx, dx)
    m_l = sp.eye(nx) - 0.5 * dt * params.D * a
    m_r = sp.eye(nx) + 0.5 * dt * params.D * a
    solve = spla.splu(m_l.tocsc()).solve

    nsteps = int(tend / dt)
    u = initial_condition(nx, init_kind)
    means = np.zeros(nsteps)
    means[0] = np.mean(u)
    for i in range(1, nsteps):
        u = solve(m_r.dot(u) + dt * reaction(u, params.p, params.q))
        means[i] = np.mean(u)

    return np.linspace(0, tend, nsteps), means
