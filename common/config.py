"""Shared configuration and constants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Params:
    """Parameters of the reaction-diffusion model u_t = D u_xx + (p + q u)(1 - u)."""

    D: float
    p: float
    q: float


# Ground-truth parameter sets used by the tasks.
PARAMS_ZAD5 = Params(D=0.1, p=0.01, q=1.0)
PARAMS_ZAD6 = Params(D=0.02, p=0.05, q=4.0)  # also used by zad7

# Sensitivity-analysis problem definition (SALib).
SALIB_PROBLEM = {
    "num_vars": 3,
    "names": ["D", "p", "q"],
    "bounds": [[0.01, 10.0], [0.001, 0.1], [1.0, 2.0]],
}

# Data-assimilation experiment setup (zad5).
ASSIM_BUDGETS = {"Low": 2, "Medium": 5, "High": 15}
PARAM_BOUNDS_ODE = {"p": (0.001, 0.1), "q": (0.1, 5.0)}
PARAM_BOUNDS_PDE = {"D": (0.01, 0.5), "p": (0.001, 0.1), "q": (0.1, 5.0)}
INITIAL_GUESS_ODE = {"p": 0.05, "q": 2.5}
INITIAL_GUESS_PDE = {"D": 0.25, "p": 0.05, "q": 2.5}

# Supermodel ensemble perturbations (zad7): (p_factor, q_factor) pairs.
SUPERMODEL_PERTURBATIONS = [(0.6, 1.4), (1.4, 0.6), (0.8, 1.1)]

TORCH_SEED = 42
NUMPY_SEED = 42


def results_dir(task: str) -> Path:
    """Return (and create) the results directory for a task, e.g. results_dir("zad6")."""
    directory = REPO_ROOT / task / "results"
    directory.mkdir(exist_ok=True, parents=True)
    return directory
