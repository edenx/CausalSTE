"""MEPS age-discontinuity figure (introduction).

Run from the repository root after the MEPS data are processed
(src.data_utils.download_meps, then src.data_utils.gen_meps --process):
    python -m src.plot_utils.meps_figure
Reads data/meps_data/meps_2019_2022_processed.csv and writes
figs/meps/meps_age_discontinuity.pdf and figs/meps/meps_age_discontinuity_gray.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot import add_panel_letters

DATA = "data/meps_data/meps_2019_2022_processed.csv"


def make_figure(df, grayscale=False, save_path="figs/meps/meps_age_discontinuity.pdf"):
    plt.style.use('seaborn-v0_8-whitegrid')
    if grayscale:
        c_ctrl, c_trt, c_line, c_cut = '#8a8a8a', '#202020', '#202020', '#202020'
    else:
        c_ctrl, c_trt, c_line, c_cut = '#3498db', '#e74c3c', '#2c3e50', '#e74c3c'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: age distribution by Medicare status
    ax = axes[0]
    treated_age = df[df['MEDICARE'] == 1]['AGE'].dropna()
    control_age = df[df['MEDICARE'] == 0]['AGE'].dropna()
    bins = np.arange(18, 100, 2)
    ax.hist(control_age, bins=bins, alpha=0.6, label='No Medicare', color=c_ctrl, density=True)
    ax.hist(treated_age, bins=bins, alpha=0.6, label='Medicare', color=c_trt, density=True,
            hatch='//' if grayscale else None)
    ax.axvline(x=65, color='black', linestyle='--', linewidth=2.5, label='Age 65 (eligibility)', alpha=0.8)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Age distribution by Medicare status', fontsize=13, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(18, 90)

    # Panel B: proportion with Medicare by age
    ax = axes[1]
    age_groups = pd.cut(df['AGE'], bins=np.arange(18, 100, 1))
    prop = df.groupby(age_groups, observed=False)['MEDICARE'].agg(['mean', 'count', 'std'])
    prop = prop[prop['count'] >= 10]
    x_pos = [i.mid for i in prop.index]
    y_mean = prop['mean']
    y_se = prop['std'] / np.sqrt(prop['count'])
    ax.plot(x_pos, y_mean, marker='o', linewidth=2, markersize=4, color=c_line, alpha=0.8)
    ax.fill_between(x_pos, y_mean - 1.96 * y_se, y_mean + 1.96 * y_se, alpha=0.2, color=c_line)
    ax.axvline(x=65, color=c_cut, linestyle='--', linewidth=2.5, label='Age 65 cutoff', alpha=0.8)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion with Medicare', fontsize=12)
    ax.set_title('Medicare coverage by age', fontsize=13, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(50, 80)

    # Panel C: mean total expenditure by age
    ax = axes[2]
    age_bins = np.arange(18, 90, 2)
    age_groups = pd.cut(df['AGE'], bins=age_bins)
    exp_trt = df[df['MEDICARE'] == 1].groupby(age_groups, observed=False)['TOTEXP'].mean()
    exp_ctrl = df[df['MEDICARE'] == 0].groupby(age_groups, observed=False)['TOTEXP'].mean()
    ax.plot([i.mid for i in exp_ctrl.index], exp_ctrl.values, marker='o', linewidth=2,
            markersize=5, label='No Medicare', color=c_ctrl, alpha=0.8)
    ax.plot([i.mid for i in exp_trt.index], exp_trt.values, marker='s', linewidth=2,
            markersize=5, label='Medicare', color=c_trt, alpha=0.8,
            linestyle='--' if grayscale else '-')
    ax.axvline(x=65, color='black', linestyle='--', linewidth=2.5, label='Age 65', alpha=0.6)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean total expenditure ($)', fontsize=12)
    ax.set_title('Healthcare expenditure by age', fontsize=13, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    add_panel_letters(axes)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figure saved to: {save_path}")


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    make_figure(df, False, "figs/meps/meps_age_discontinuity.pdf")
    make_figure(df, True, "figs/meps/meps_age_discontinuity_gray.pdf")
