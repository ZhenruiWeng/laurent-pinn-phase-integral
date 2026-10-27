# Copyright (c) 2026 Zhenrui Weng. Licensed under the MIT License.
"""Generate big, clean single-panel fit plots for all benchmarks."""
import numpy as np
import math
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from laurent_pinn_learnable_xs import (
    LaurentPINN, BENCHMARKS, train, evaluate, device, seed_everything
)


def break_curve(x, y):
    """Insert NaN at big jumps in x so plot lines break at the asymptote."""
    if len(x) < 2:
        return x, y
    dx = np.diff(x)
    jumps = np.where(dx > 5 * np.median(dx))[0]
    xl, yl = list(x), list(y)
    for j in reversed(jumps):
        xl.insert(j + 1, float(x[j + 1]))
        yl.insert(j + 1, float('nan'))
    return np.array(xl), np.array(yl)


def big_fit_plot(bench_name, xs_init=None, epochs=6000, out_path=None):
    bench = BENCHMARKS[bench_name]
    if xs_init is None:
        xs_init = bench['inits'][1]  # middle init
    if out_path is None:
        out_path = f'figs/{bench_name}_fit_big.png'

    seed_everything(1314)
    model = LaurentPINN(
        window=bench['window'], search=bench['search'],
        xs_init=xs_init, A_init=bench['A_true'],
        alpha=bench['alpha_true'], soft_pole=True,
        x_bc=bench['x_bc'], u_bc=bench['u_bc']
    ).to(device)

    hist = train(model, bench, n_epochs=epochs, tag=f'bigfit_{bench_name}')
    g, u_pred, u_exact, m = evaluate(model, bench)

    g_b, ue_b = break_curve(g, u_exact)
    _, up_b = break_curve(g, u_pred)
    err = np.abs(u_pred - u_exact)
    g_err = g  # error on the original grid (no NaN breaks for semilogy)

    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(12, 9),
        gridspec_kw={'height_ratios': [3, 1.2]},
        sharex=True,
    )

    # --- top: fit curve ---
    # Black solid (exact) first, red dashed (PINN) on TOP. Red is noticeably
    # thicker so it forms a visible "outline" around the black exact curve
    # even when the two coincide nearly perfectly.
    ax.plot(g_b, ue_b, 'k-', lw=1.5, label='Exact (black solid)', zorder=2)
    ax.plot(g_b, up_b, color='#ff2222', lw=3.2, dashes=(7, 4),
            label='Laurent PINN (red dashed)', zorder=3)

    xs_true = bench.get('xs_true')
    if xs_true is not None:
        label = 'True pole $x_s^*$ = {:.4f}'.format(xs_true)
        ax.axvline(xs_true, color='gray', ls=':', lw=1.5, alpha=0.7, label=label)
        xs_hat = m['xs_hat']
        label2 = 'Learned pole $\hat x_s$ = {:.4f}'.format(xs_hat)
        ax.axvline(xs_hat, color='tab:red', ls='--', lw=1.2, alpha=0.6, label=label2)

    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_ylabel('u(x)', fontsize=13)

    # auto y-lim: use 10x median magnitude to show the S-shape
    med = np.median(np.abs(u_exact))
    cap = max(med * 10, 5.0)
    ax.set_ylim(-cap, cap)

    info = 'xs_err = {:.2e}   A_err = {:.2e}   MAE = {:.2e}'.format(
        m.get('xs_err', float('nan')),
        m.get('A_err', float('nan')),
        m['mae'],
    )
    title = bench_name.upper() + r' — Laurent PINN fit' + '\n' + info
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(alpha=0.3)

    # --- bottom: pointwise error ---
    ax_err.semilogy(g_err, err, 'tab:red', lw=1.2)
    ax_err.fill_between(g_err, err, 1e-8, alpha=0.15, color='tab:red')
    ax_err.set_ylabel('|error|', fontsize=12)
    ax_err.set_xlabel('x', fontsize=13)
    ax_err.set_title('Pointwise absolute error (log scale)', fontsize=11)
    ax_err.grid(alpha=0.3, which='both')
    ax_err.set_ylim(bottom=max(err.min() * 0.1, 1e-7))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('[{}] saved -> {}  xs_hat={:.6f}  xs_err={:.2e}  mae={:.2e}'.format(
        bench_name, out_path, m.get('xs_hat', float('nan')),
        m.get('xs_err', float('nan')), m['mae']))
    return m


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--bench', nargs='+', default=None,
                   help='benchmarks to plot (default: all)')
    p.add_argument('--epochs', type=int, default=6000)
    args = p.parse_args()

    names = args.bench if args.bench else list(BENCHMARKS.keys())
    for name in names:
        try:
            big_fit_plot(name, epochs=args.epochs)
        except Exception as e:
            print('[{}] FAILED: {}'.format(name, e))
