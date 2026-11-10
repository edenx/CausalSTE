"""Density-ratio illustration (overlap examples figure in the appendix).

Run from the repository root:  python -m src.plot_utils.examples_plot
Writes figs/density_ratio_plots.pdf (colour) and figs/density_ratio_plots_gray.pdf
(greyscale for print). Panels carry the letters A to H in the upper-left corner.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

from .plot import add_panel_letters


def make_figure(grayscale=False, save_path="figs/density_ratio_plots.pdf"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    if grayscale:
        color_p0, color_p1, color_ratio = '#202020', '#8a8a8a', '#000000'
        ls_p0, ls_p1 = '-', '--'
        color_neg, color_pos = '#202020', '#8a8a8a'
        color_warn = 'black'
    else:
        color_p0, color_p1, color_ratio = '#2E86AB', '#E94F37', '#7B2D8E'
        ls_p0, ls_p1 = '-', '-'
        color_neg, color_pos = 'green', 'orange'
        color_warn = 'red'

    def densities(ax, x, p0, p1, lab0, lab1, title, step=None):
        kw = {'step': step} if step else {}
        ax.fill_between(x, p0, alpha=0.3, color=color_p0, **kw)
        ax.plot(x, p0, color=color_p0, lw=2, ls=ls_p0, label=lab0)
        ax.fill_between(x, p1, alpha=0.3, color=color_p1, **kw)
        ax.plot(x, p1, color=color_p1, lw=2, ls=ls_p1, label=lab1)
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right', fontsize=8)

    def log_ratio(ax, x, lr):
        ax.fill_between(x, 0, lr, where=(lr > 0), alpha=0.3, color=color_ratio)
        ax.fill_between(x, lr, 0, where=(lr < 0), alpha=0.3, color=color_ratio)
        ax.plot(x, lr, color=color_ratio, lw=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\log \frac{p_X^1}{p_X^0}$')

    # Case 1: Normal distributions, P_X^0 = N(-10, 2^2), P_X^1 = N(10, 5^2)
    x1 = np.linspace(-20, 30, 1000)
    p0_1 = stats.norm.pdf(x1, loc=-10, scale=2)
    p1_1 = stats.norm.pdf(x1, loc=10, scale=5)
    lr1 = np.log(2 / 5) - (x1 - 10) ** 2 / (2 * 25) + (x1 + 10) ** 2 / (2 * 4)
    densities(axes[0, 0], x1, p0_1, p1_1,
              r'$p_X^0 = \mathcal{N}(-10, 2^2)$', r'$p_X^1 = \mathcal{N}(10, 5^2)$', 'Normal')
    log_ratio(axes[1, 0], x1, lr1)

    # Case 2: Beta distributions, P_X^0 = Beta(0.5, 2), P_X^1 = Beta(1.3, 0.5)
    x2 = np.linspace(0.001, 0.999, 1000)
    p0_2 = stats.beta.pdf(x2, 0.5, 2)
    p1_2 = stats.beta.pdf(x2, 1.3, 0.5)
    densities(axes[0, 1], x2, p0_2, p1_2,
              r'$p_X^0 = \mathrm{Beta}(0.5, 2)$', r'$p_X^1 = \mathrm{Beta}(1.3, 0.5)$', 'Beta')
    axes[0, 1].set_ylim(0, 5)
    log_ratio(axes[1, 1], x2, np.log(p1_2 / p0_2))

    # Case 3: Uniform distributions, P_X^0 = Unif[-1, 1], P_X^1 = Unif[0, 5]
    x3 = np.linspace(-2, 6, 1000)
    p0_3 = stats.uniform.pdf(x3, loc=-1, scale=2)
    p1_3 = stats.uniform.pdf(x3, loc=0, scale=5)
    densities(axes[0, 2], x3, p0_3, p1_3,
              r'$p_X^0 = \mathrm{Unif}[-1, 1]$', r'$p_X^1 = \mathrm{Unif}[0, 5]$', 'Uniform', step='mid')
    axes[0, 2].set_ylim(0, 0.7)
    ax = axes[1, 2]
    ax.hlines(np.log(2 / 5), 0, 1, color=color_ratio, lw=3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.annotate(r'$-\infty$', xy=(-0.5, -4), fontsize=10, ha='center', color=color_neg, fontweight='bold')
    ax.annotate('', xy=(-0.5, -5.5), xytext=(-0.5, -3),
                arrowprops=dict(arrowstyle='->', color=color_neg, lw=2))
    ax.annotate(r'$+\infty$', xy=(3, 4), fontsize=10, ha='center', color=color_pos, fontweight='bold')
    ax.annotate('', xy=(3, 5.5), xytext=(3, 3),
                arrowprops=dict(arrowstyle='->', color=color_pos, lw=2))
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\log \frac{p_X^1}{p_X^0}$')
    ax.set_xlim(-2, 6)
    ax.set_ylim(-6, 6)
    for v in (-1, 0, 1, 5):
        ax.axvline(x=v, color='gray', linestyle=':', alpha=0.5)

    # Case 4: discrete point masses on disjoint supports
    x_points = np.array([-3, -1, 1, 3, 5])
    y_points = np.array([-2, 0, 2, 4, 6])
    weights_0 = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    weights_1 = np.array([0.15, 0.25, 0.3, 0.2, 0.1])
    ax = axes[0, 3]
    m0, s0, _ = ax.stem(x_points, weights_0, basefmt=' ', linefmt='-', markerfmt='o')
    plt.setp(s0, color=color_p0, linewidth=2)
    plt.setp(m0, color=color_p0, markersize=8)
    m1, s1, _ = ax.stem(y_points, weights_1, basefmt=' ', linefmt='-', markerfmt='s')
    plt.setp(s1, color=color_p1, linewidth=2)
    plt.setp(m1, color=color_p1, markersize=8)
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Mass')
    ax.set_ylim(0, 0.45)
    ax.set_xlim(-5, 8)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color=color_p0, lw=2, markersize=6,
               label=r'$p_X^0 = \sum_i w_i^0 \delta_{x_i}$'),
        Line2D([0], [0], marker='s', color=color_p1, lw=2, markersize=6,
               label=r'$p_X^1 = \sum_j w_j^1 \delta_{y_j}$')],
        loc='upper right', fontsize=8)
    ax = axes[1, 3]
    ax.set_xlim(-5, 8)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\log \frac{p_X^1}{p_X^0}$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, r'$\mathbf{Undefined}$' + '\n\nDisjoint supports:\n'
            + r'$\mathrm{supp}(p_X^0) \cap \mathrm{supp}(p_X^1) = \emptyset$',
            ha='center', va='center', transform=ax.transAxes, fontsize=11, color=color_warn,
            bbox=dict(boxstyle='round', facecolor='lightyellow' if not grayscale else '#f0f0f0',
                      alpha=0.9, edgecolor=color_warn, lw=2))

    add_panel_letters(axes)
    plt.tight_layout()
    plt.subplots_adjust(left=0.06)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✓ Figure saved to: {save_path}")


if __name__ == "__main__":
    make_figure(grayscale=False, save_path="figs/density_ratio_plots.pdf")
    make_figure(grayscale=True, save_path="figs/density_ratio_plots_gray.pdf")
