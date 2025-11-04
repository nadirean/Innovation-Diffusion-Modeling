"""Numerical solvers for 1D reaction-diffusion model

Model: u_t = D*u_xx + R(u),  R(u) = (p + q*u)*(1-u)

Solvers implemented:
- explicit: forward Euler in time, 2nd-order central in space
- semi_implicit: diffusion implicit (backward Euler for diffusion), reaction explicit
- crank_nicolson: diffusion Crank-Nicolson, reaction explicit

Neumann BCs implemented via ghost points (zero-gradient) by reflecting boundary values.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import perf_counter
from typing import Tuple


def reaction(u: np.ndarray, p: float, q: float) -> np.ndarray:
    return (p + q * u) * (1 - u)


def _neumann_laplacian(nx: int, dx: float) -> sp.csr_matrix:
    # second-derivative operator with Neumann BCs on [0,1]
    main = -2.0 * np.ones(nx)
    off = 1.0 * np.ones(nx - 1)
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], format='csr')
    # Neumann adjustments: for second derivative, reflect at boundaries
    A[0, 0] = -1.0
    A[-1, -1] = -1.0
    return A / (dx * dx)


def initial_condition(nx: int, dx: float, kind: str = 'gaussian') -> np.ndarray:
    x = np.linspace(0, 1, nx)
    if kind == 'gaussian':
        u0 = 0.01 * np.ones_like(x)
        u0 += 0.5 * np.exp(-((x - 0.5) ** 2) / (2 * 0.02 ** 2))
        return u0
    elif kind == 'localized':
        u0 = 0.0 * x
        mask = np.abs(x - 0.5) < 0.05
        u0[mask] = 0.5
        u0 += 1e-3
        return u0
    elif kind == 'small_random':
        rng = np.random.default_rng(0)
        return 1e-3 + 0.01 * rng.random(nx)
    else:
        raise ValueError('unknown init kind')


def run_simulation(
    solver: str,
    nx: int = 100,
    dt: float = 1e-4,
    tend: float = 0.1,
    D: float = 0.01,
    p: float = 0.01,
    q: float = 1.0,
    init_kind: str = 'gaussian',
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run simulation and return x, u, elapsed_time"""
    dx = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx)
    u = initial_condition(nx, dx, kind=init_kind)

    A = _neumann_laplacian(nx, dx)

    nsteps = int(np.ceil(tend / dt))

    t0 = perf_counter()

    if solver == 'explicit':
        for n in range(nsteps):
            Lu = A.dot(u)
            u = u + dt * (D * Lu + reaction(u, p, q))
    elif solver == 'semi-implicit':
        M = sp.eye(nx) - dt * D * A
        for n in range(nsteps):
            Ru = reaction(u, p, q)
            rhs = u + dt * Ru
            u = spla.spsolve(M, rhs)
    elif solver in ('crank-nicolson', 'cn'):
        M_l = sp.eye(nx) - 0.5 * dt * D * A
        M_r = sp.eye(nx) + 0.5 * dt * D * A
        for n in range(nsteps):
            Ru = reaction(u, p, q)
            rhs = M_r.dot(u) + dt * Ru
            u = spla.spsolve(M_l, rhs)
    else:
        raise ValueError('unknown solver')

    elapsed = perf_counter() - t0
    return x, u, elapsed


if __name__ == '__main__':
    # quick smoke test
    x, u, t = run_simulation('explicit', nx=50, dt=1e-4, tend=0.01)
    print('Done test, elapsed=', t)
