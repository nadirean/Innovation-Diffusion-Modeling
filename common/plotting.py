"""Shared plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def savefig(path: Path, dpi: int = 150) -> None:
    """Save the current figure (parent directory is created)."""
    path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_noise_slice(
    x: NDArray[np.float64],
    u_clean: NDArray[np.float64],
    u_noisy: NDArray[np.float64],
    out: Path,
) -> None:
    """Clean vs noisy measurement preview at t=0."""
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_clean[:, 0], label="clean t=0")
    plt.plot(x, u_noisy[:, 0], label="noisy t=0", alpha=0.7)
    plt.title("Measurement slice (t=0)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    savefig(out)
    plt.show()
