"""Data assimilation methods: ABC rejection sampling and 3D-Var style optimization (task 5)."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .model import neumann_laplacian, reaction

ModelFunc = Callable[[dict[str, float], NDArray[np.float64]], NDArray[np.float64]]


def run_pde_model(
    params: dict[str, float],
    t_eval: NDArray[np.float64],
    nx: int,
    dt: float,
    defaults: dict[str, float],
) -> NDArray[np.float64]:
    """Full-field Crank-Nicolson solve with a narrow-gaussian initial state (time-major output)."""
    p = {**defaults, **params}
    dx = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx)
    u = np.exp(-100.0 * (x - 0.5) ** 2)

    a = neumann_laplacian(nx, dx)
    m_l = sp.eye(nx) - 0.5 * dt * p["D"] * a
    m_r = sp.eye(nx) + 0.5 * dt * p["D"] * a
    solve = spla.splu(m_l.tocsc()).solve

    t_eval_sorted = np.sort(t_eval)
    results: list[NDArray[np.float64]] = []
    idx = 0
    if idx < len(t_eval_sorted) and t_eval_sorted[idx] == 0.0:
        results.append(u.copy())
        idx += 1

    t = 0.0
    for _ in range(int(np.ceil(t_eval_sorted[-1] / dt))):
        u = solve(m_r.dot(u) + dt * reaction(u, p["p"], p["q"]))
        t += dt
        while idx < len(t_eval_sorted) and t >= t_eval_sorted[idx]:
            results.append(u.copy())
            idx += 1

    return np.asarray(results)


def run_ode_model(
    params: dict[str, float],
    t_eval: NDArray[np.float64],
    nx: int,
    defaults: dict[str, float],
) -> NDArray[np.float64]:
    """Solve the ODE surrogate from the mean narrow-gaussian initial state."""
    p = {**defaults, **params}
    x = np.linspace(0, 1, nx)
    u0 = float(np.mean(np.exp(-100.0 * (x - 0.5) ** 2)))
    sol = solve_ivp(
        lambda t, U: (p["p"] + p["q"] * U) * (1.0 - U),
        [0.0, t_eval[-1]],
        [u0],
        t_eval=t_eval,
    )
    return sol.y.T


def run_abc(
    model_func: ModelFunc,
    obs_data: NDArray[np.float64],
    t_eval: NDArray[np.float64],
    param_bounds: dict[str, tuple[float, float]],
    budget_seconds: float,
) -> tuple[dict[str, float] | None, float | None]:
    """ABC rejection sampling: uniform priors, keep top 5% by MSE, average them."""
    start = time.time()
    samples: list[dict[str, float]] = []
    errors: list[float] = []
    while time.time() - start < budget_seconds:
        sample = {k: float(np.random.uniform(*b)) for k, b in param_bounds.items()}
        try:
            errors.append(float(np.mean((model_func(sample, t_eval) - obs_data) ** 2)))
        except Exception:
            continue
        samples.append(sample)
    if not errors:
        return None, None
    errors_arr = np.asarray(errors)
    keep = errors_arr <= np.percentile(errors_arr, 5)
    best = [samples[i] for i in np.nonzero(keep)[0]]
    estimated = {k: float(np.mean([s[k] for s in best])) for k in param_bounds}
    return estimated, float(np.min(errors_arr))


def run_variational(
    model_func: ModelFunc,
    obs_data: NDArray[np.float64],
    t_eval: NDArray[np.float64],
    param_bounds: dict[str, tuple[float, float]],
    budget_seconds: float,
    initial_guess: dict[str, float],
) -> tuple[dict[str, float], float]:
    """3D-Var style fitting: L-BFGS-B on MSE, keeping the best point under a hard time budget."""
    start = time.time()
    keys = list(param_bounds.keys())
    x0 = [initial_guess[k] for k in keys]
    bounds = [param_bounds[k] for k in keys]
    best_x, best_mse = x0, float("inf")

    def cost(x: list[float]) -> float:
        nonlocal best_x, best_mse
        if time.time() - start > budget_seconds:
            raise TimeoutError
        try:
            mse = float(np.mean((model_func(dict(zip(keys, x)), t_eval) - obs_data) ** 2))
        except Exception:
            return 1e9
        if mse < best_mse:
            best_x, best_mse = x, mse
        return mse

    try:
        minimize(cost, x0, bounds=bounds, method="L-BFGS-B")
    except TimeoutError:
        pass
    print(f"Variational finished. Time: {time.time() - start:.2f}s, Best MSE: {best_mse:.6f}")
    return dict(zip(keys, best_x)), best_mse


def evaluate_prediction(
    model_func: ModelFunc,
    params: dict[str, float],
    t_full: NDArray[np.float64],
    u_true: NDArray[np.float64],
    assim_range: tuple[float, float],
) -> tuple[NDArray[np.float64], float, float, float]:
    """Predicted trajectory plus total / in-window / out-of-window RMSE."""
    u_pred = model_func(params, t_full)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mask = (t_full >= assim_range[0]) & (t_full <= assim_range[1])
    return u_pred, rmse(u_pred, u_true), rmse(u_pred[mask], u_true[mask]), rmse(u_pred[~mask], u_true[~mask])
