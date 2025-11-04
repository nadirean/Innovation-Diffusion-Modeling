"""Run parameter sweep experiments and save timings/outputs.

Example usage:
    python3 experiment.py

By default the script runs a small grid of experiments. Use CLI args to change solver, nx, dt, tend.
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solvers import run_simulation


OUTDIR = Path(__file__).parent / 'results'
OUTDIR.mkdir(exist_ok=True)


def run_default_suite():
    solvers = ['explicit', 'semi-implicit', 'crank-nicolson']
    nx_list = [50, 100, 200]
    dt_list = [1e-4, 5e-5]
    params = [
        {'D': 0.01, 'p': 0.01, 'q': 1.0},
        {'D': 0.1, 'p': 0.001, 'q': 5.0},
        {'D': 1.0, 'p': 0.1, 'q': 10.0},
    ]

    rows = []
    for s in solvers:
        for nx in nx_list:
            for dt in dt_list:
                for prm in params:
                    try:
                        x, u, elapsed = run_simulation(
                            solver=s,
                            nx=nx,
                            dt=dt,
                            tend=0.05,
                            D=prm['D'],
                            p=prm['p'],
                            q=prm['q'],
                            init_kind='gaussian',
                        )
                    except Exception as e:
                        elapsed = float('nan')
                        print('Run failed:', e)

                    name = f"{s}_nx{nx}_dt{dt}_D{prm['D']}_p{prm['p']}_q{prm['q']}"
                    rows.append({
                        'name': name,
                        'solver': s,
                        'nx': nx,
                        'dt': dt,
                        'D': prm['D'],
                        'p': prm['p'],
                        'q': prm['q'],
                        'elapsed': elapsed,
                    })
                    # save a representative final profile
                    try:
                        plt.figure()
                        plt.plot(x, u)
                        plt.xlabel('x')
                        plt.ylabel('u')
                        plt.title(name)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(OUTDIR / (name + '.png'))
                        plt.close()
                    except Exception:
                        pass

    df = pd.DataFrame(rows)
    csvp = OUTDIR / 'timings.csv'
    df.to_csv(csvp, index=False)
    print('Saved timings to', csvp)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', default=None)
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--tend', type=float, default=0.05)
    parser.add_argument('--D', type=float, default=0.01)
    parser.add_argument('--p', type=float, default=0.01)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--init', default='gaussian')
    args = parser.parse_args()

    if args.solver is None:
        run_default_suite()
        return

    x, u, elapsed = run_simulation(
        solver=args.solver,
        nx=args.nx,
        dt=args.dt,
        tend=args.tend,
        D=args.D,
        p=args.p,
        q=args.q,
        init_kind=args.init,
    )
    out = {
        'solver': args.solver,
        'nx': args.nx,
        'dt': args.dt,
        'tend': args.tend,
        'D': args.D,
        'p': args.p,
        'q': args.q,
        'elapsed': elapsed,
    }
    outpath = OUTDIR / f"single_{args.solver}_nx{args.nx}.json"
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved single run info to', outpath)


if __name__ == '__main__':
    cli()
