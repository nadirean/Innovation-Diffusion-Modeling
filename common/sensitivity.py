"""Sensitivity analysis runners and plots (task 4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .config import SALIB_PROBLEM
from .model import initial_condition
from .solvers import run_simulation


def evaluate_pde(X: NDArray[np.float64], nx: int, dt: float, tend: float) -> NDArray[np.float64]:
    """Mean final profile u(T) of the PDE for each parameter row (D, p, q) in X."""
    outputs = np.zeros(X.shape[0])
    for i, (D, p, q) in enumerate(X):
        _, u_final, _ = run_simulation("crank-nicolson", nx=nx, dt=dt, tend=tend, D=D, p=p, q=q)
        outputs[i] = np.mean(u_final)
        if (i + 1) % 20 == 0:
            print(f"PDE evaluation: {i + 1}/{X.shape[0]}")
    return outputs


def evaluate_ode(X: NDArray[np.float64], nx: int, tend: float) -> NDArray[np.float64]:
    """Final value U(T) of the ODE surrogate for each parameter row (D, p, q) in X."""
    u0 = float(np.mean(initial_condition(nx, "gaussian")))
    outputs = np.zeros(X.shape[0])
    for i, (_, p, q) in enumerate(X):
        sol = solve_ivp(lambda t, U: (p + q * U) * (1.0 - U), [0.0, tend], [u0], dense_output=True)
        outputs[i] = sol.sol(tend)[0]
    return outputs


def plot_morris(results: dict, model_name: str, tag: str, fig_dir: Path) -> None:
    """Scatter mu_star vs sigma for a Morris analysis."""
    names = SALIB_PROBLEM["names"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(names):
        plt.scatter(results["mu_star"][i], results["sigma"][i], label=name, s=100, alpha=0.7)
    plt.xlabel("mu* (overall influence)")
    plt.ylabel("sigma (interactions / nonlinearity)")
    plt.title(f"Morris analysis - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.5, -0.05, "Top-right parameters are the most important and most nonlinear.", ha="center", fontsize=10)
    plt.savefig(fig_dir / f"morris_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_sobol(results: dict, model_name: str, tag: str, fig_dir: Path) -> None:
    """Stacked bar chart of S1 and interaction components for a Sobol analysis."""
    names = SALIB_PROBLEM["names"]
    fig_dir.mkdir(exist_ok=True, parents=True)
    s1 = results["S1"]
    st = results["ST"]
    index = np.arange(len(names))
    plt.figure(figsize=(10, 6))
    plt.bar(index - 0.175, s1, 0.35, label="S1 (direct influence)")
    plt.bar(index + 0.175, st - s1, 0.35, bottom=s1, label="Interactions")
    plt.ylabel("Sobol indices")
    plt.title(f"Sobol analysis - {model_name}")
    plt.xticks(index, names)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(fig_dir / f"sobol_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()
