"""Numerical solvers for the 1D reaction-diffusion model.

Model: u_t = D*u_xx + R(u),  R(u) = (p + q*u)*(1-u)

Solvers:
- explicit: forward Euler in time, 2nd-order central in space (CFL-limited)
- semi-implicit: implicit diffusion (backward Euler), explicit reaction
- crank-nicolson: implicit Crank-Nicolson diffusion, explicit reaction
"""
from __future__ import annotations

from time import perf_counter
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .model import initial_condition, neumann_laplacian, reaction


def run_simulation(
    solver: str,
    nx: int = 100,
    dt: float = 1e-4,
    tend: float = 0.1,
    D: float = 0.01,
    p: float = 0.01,
    q: float = 1.0,
    init_kind: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run a simulation and return (x, u_final, elapsed_time).

    Raises ValueError when the explicit solver violates the CFL condition
    (D*dt/dx^2 > 0.5) instead of silently producing NaN.
    """
    dx = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx)
    u = initial_condition(nx, init_kind)

    a = neumann_laplacian(nx, dx)
    nsteps = int(np.ceil(tend / dt))

    t0 = perf_counter()

    if solver == "explicit":
        r = D * dt / dx**2
        if r > 0.5:
            raise ValueError(
                f"CFL condition violated for explicit solver: "
                f"D*dt/dx^2 = {r:.3f} > 0.5. Use a smaller dt or a larger nx."
            )
        for _ in range(nsteps):
            u = u + dt * (D * a.dot(u) + reaction(u, p, q))
    elif solver == "semi-implicit":
        solve = spla.splu((sp.eye(nx) - dt * D * a).tocsc()).solve
        for _ in range(nsteps):
            u = solve(u + dt * reaction(u, p, q))
    elif solver in ("crank-nicolson", "cn"):
        m_l = sp.eye(nx) - 0.5 * dt * D * a
        m_r = sp.eye(nx) + 0.5 * dt * D * a
        solve = spla.splu(m_l.tocsc()).solve
        for _ in range(nsteps):
            u = solve(m_r.dot(u) + dt * reaction(u, p, q))
    else:
        raise ValueError(f"unknown solver: {solver}")

    return x, u, perf_counter() - t0


if __name__ == "__main__":
    x, u, t = run_simulation("explicit", nx=50, dt=1e-4, tend=0.01)
    print("Done test, elapsed=", t)
