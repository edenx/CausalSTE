"""Covariate distributions of the two simulation designs (appendix figure).

Run from the repository root:  python -m src.plot_utils.sim_density_figure
Writes figs/sim_covariate_distributions.pdf and figs/sim_covariate_distributions_gray.pdf.

Panels A to D: the Beta design, X | A=1 ~ Beta(1.3, 1.55 - 0.1 s; 50, -25) and
X | A=0 ~ Beta(1.55 - 0.1 s, 2; 50, -25), at s = 0 and s = 13. Panels E to H: the
Gaussian design, X | A=1 ~ N(-s, 5^2+1) and X | A=0 ~ N(s, 5^2+1), at s = 0 and s = 13
(the code adds a standard normal to each N(., 5^2) coordinate, hence variance 26).
Left column of each block: the two densities; right column: the density ratios
in both directions on a log scale.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .plot import add_panel_letters

S_VALUES = (0, 13)
SCALE, LOC = 50.0, -25.0  # Beta(a, b; 50, -25) = 50 * Beta(a, b) - 25
VAR_G = 5.0 ** 2 + 1.0       # N(., 5^2) plus a standard normal per coordinate
SD_G = VAR_G ** 0.5


def beta_params(s):
    a1, b1 = 1.3, 1.55 - 0.1 * s      # treated
    a0, b0 = 1.55 - 0.1 * s, 2.0      # control
    return (a0, b0), (a1, b1)


def beta_pdf(x, a, b):
    return stats.beta.pdf((x - LOC) / SCALE, a, b) / SCALE


def make_figure(grayscale=False, save_path="figs/sim_covariate_distributions.pdf"):
    plt.style.use('seaborn-v0_8-whitegrid')
    if grayscale:
        c0, c1, ls0, ls1 = '#202020', '#8a8a8a', '-', '--'
    else:
        c0, c1, ls0, ls1 = 'blue', 'red', '-', '-'

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))

    for row, s in enumerate(S_VALUES):
        # --- Beta design (columns 0 and 1) ---
        (a0, b0), (a1, b1) = beta_params(s)
        x = np.linspace(LOC + 1e-3, LOC + SCALE - 1e-3, 2000)
        p0, p1 = beta_pdf(x, a0, b0), beta_pdf(x, a1, b1)
        ax = axes[row, 0]
        ax.plot(x, p0, color=c0, ls=ls0, lw=2, label=rf'$P_X^0 = \mathrm{{Beta}}({a0:.2f}, {b0:.2f};\,50,-25)$')
        ax.fill_between(x, p0, color=c0, alpha=0.2)
        ax.plot(x, p1, color=c1, ls=ls1, lw=2, label=rf'$P_X^1 = \mathrm{{Beta}}({a1:.2f}, {b1:.2f};\,50,-25)$')
        ax.fill_between(x, p1, color=c1, alpha=0.2)
        ax.set_ylabel(f'Density, $s={s}$')
        ax.set_xlabel('Value')
        ax.set_ylim(0, min(np.percentile(np.r_[p0, p1], 99.5) * 1.3, 0.2))
        ax.legend(loc='upper center', fontsize=8)
        if row == 0:
            ax.set_title('Beta design: densities', fontweight='bold')

        ax = axes[row, 1]
        ax.plot(x, p0 / p1, color=c0, ls=ls0, lw=2, label=r'$p_X^0 / p_X^1$')
        ax.plot(x, p1 / p0, color=c1, ls=ls1, lw=2, label=r'$p_X^1 / p_X^0$')
        ax.set_yscale('log')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density ratio (log scale)')
        ax.legend(loc='upper center', fontsize=8)
        if row == 0:
            ax.set_title('Beta design: density ratios', fontweight='bold')

        # --- Gaussian design (columns 2 and 3) ---
        xg = np.linspace(-30, 30, 2000)
        g0, g1 = stats.norm.pdf(xg, s, SD_G), stats.norm.pdf(xg, -s, SD_G)
        ax = axes[row, 2]
        ax.plot(xg, g0, color=c0, ls=ls0, lw=2, label=rf'$P_X^0 = N({s}, 5^2+1)$')
        ax.fill_between(xg, g0, color=c0, alpha=0.2)
        ax.plot(xg, g1, color=c1, ls=ls1, lw=2, label=rf'$P_X^1 = N({-s}, 5^2+1)$')
        ax.fill_between(xg, g1, color=c1, alpha=0.2)
        ax.set_ylabel(f'Density, $s={s}$')
        ax.set_xlabel('Value')
        ax.legend(loc='upper center', fontsize=8)
        if row == 0:
            ax.set_title('Gaussian design: densities', fontweight='bold')

        ax = axes[row, 3]
        # analytic ratio exp(((x+s)^2 - (x-s)^2)/(2*VAR_G)) = exp(2 s x / VAR_G)
        ax.plot(xg, np.exp(-2 * s * xg / VAR_G), color=c0, ls=ls0, lw=2, label=r'$p_X^0 / p_X^1$')
        ax.plot(xg, np.exp(2 * s * xg / VAR_G), color=c1, ls=ls1, lw=2, label=r'$p_X^1 / p_X^0$')
        ax.set_yscale('log')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density ratio (log scale)')
        ax.legend(loc='upper center', fontsize=8)
        if row == 0:
            ax.set_title('Gaussian design: density ratios', fontweight='bold')

    add_panel_letters(axes)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figure saved to: {save_path}")


if __name__ == "__main__":
    make_figure(False, "figs/sim_covariate_distributions.pdf")
    make_figure(True, "figs/sim_covariate_distributions_gray.pdf")
