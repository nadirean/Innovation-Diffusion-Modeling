"""Helpers to plot results saved by experiment.py"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_timings(csvpath: str | Path):
    df = pd.read_csv(csvpath)
    fig, ax = plt.subplots()
    for solver, g in df.groupby('solver'):
        ax.plot(g['nx'], g['elapsed'], marker='o', linestyle='-', label=solver)
    ax.set_xlabel('nx')
    ax.set_ylabel('elapsed (s)')
    ax.set_title('Timing vs nx')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    out = Path(csvpath).with_suffix('.png')
    plt.savefig(out)
    print('Saved plot to', out)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: plot_results.py results/timings.csv')
    else:
        plot_timings(sys.argv[1])
