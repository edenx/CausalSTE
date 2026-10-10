import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# Set up the figure with 2 rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Colors
color_p0 = '#2E86AB'  # Blue
color_p1 = '#E94F37'  # Red
color_ratio = '#7B2D8E'  # Purple

# ============== Case 1: Normal distributions ==============
# P_X^0 = N(-10, 2^2), P_X^1 = N(10, 5^2)
x1 = np.linspace(-20, 30, 1000)
p0_1 = stats.norm.pdf(x1, loc=-10, scale=2)
p1_1 = stats.norm.pdf(x1, loc=10, scale=5)

# Analytical log ratio for Gaussians
log_ratio_1 = (np.log(2/5) 
               - (x1 - 10)**2 / (2 * 25) 
               + (x1 + 10)**2 / (2 * 4))

# Row 1: Density plot
axes[0, 0].fill_between(x1, p0_1, alpha=0.3, color=color_p0)
axes[0, 0].plot(x1, p0_1, color=color_p0, lw=2, label=r'$p_X^0 = \mathcal{N}(-10, 2^2)$')
axes[0, 0].fill_between(x1, p1_1, alpha=0.3, color=color_p1)
axes[0, 0].plot(x1, p1_1, color=color_p1, lw=2, label=r'$p_X^1 = \mathcal{N}(10, 5^2)$')
axes[0, 0].set_title('(1) Normal', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend(loc='upper right', fontsize=8)

# Row 2: Log density ratio plot
axes[1, 0].fill_between(x1, 0, log_ratio_1, where=(log_ratio_1 > 0), alpha=0.3, color=color_ratio)
axes[1, 0].fill_between(x1, log_ratio_1, 0, where=(log_ratio_1 < 0), alpha=0.3, color=color_ratio)
axes[1, 0].plot(x1, log_ratio_1, color=color_ratio, lw=2)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel(r'$\log \frac{p_X^1}{p_X^0}$')

# ============== Case 2: Beta distributions ==============
# P_X^0 = Beta(0.5, 2), P_X^1 = Beta(1.3, 0.5)
x2 = np.linspace(0.001, 0.999, 1000)
p0_2 = stats.beta.pdf(x2, 0.5, 2)
p1_2 = stats.beta.pdf(x2, 1.3, 0.5)
ratio_2 = p1_2 / p0_2

# Row 1: Density plot
axes[0, 1].fill_between(x2, p0_2, alpha=0.3, color=color_p0)
axes[0, 1].plot(x2, p0_2, color=color_p0, lw=2, label=r'$p_X^0 = \mathrm{Beta}(0.5, 2)$')
axes[0, 1].fill_between(x2, p1_2, alpha=0.3, color=color_p1)
axes[0, 1].plot(x2, p1_2, color=color_p1, lw=2, label=r'$p_X^1 = \mathrm{Beta}(1.3, 0.5)$')
axes[0, 1].set_title('(2) Beta', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_ylim(0, 5)
axes[0, 1].legend(loc='upper right', fontsize=8)

# Row 2: Log density ratio plot
log_ratio_2 = np.log(ratio_2)
axes[1, 1].fill_between(x2, 0, log_ratio_2, where=(log_ratio_2 > 0), alpha=0.3, color=color_ratio)
axes[1, 1].fill_between(x2, log_ratio_2, 0, where=(log_ratio_2 < 0), alpha=0.3, color=color_ratio)
axes[1, 1].plot(x2, log_ratio_2, color=color_ratio, lw=2)
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel(r'$\log \frac{p_X^1}{p_X^0}$')

# ============== Case 3: Uniform distributions ==============
# P_X^0 = Unif[-1, 1], P_X^1 = Unif[0, 5]
x3 = np.linspace(-2, 6, 1000)
p0_3 = stats.uniform.pdf(x3, loc=-1, scale=2)  # Unif[-1, 1]
p1_3 = stats.uniform.pdf(x3, loc=0, scale=5)   # Unif[0, 5]

# Row 1: Density plot
axes[0, 2].fill_between(x3, p0_3, alpha=0.3, color=color_p0, step='mid')
axes[0, 2].plot(x3, p0_3, color=color_p0, lw=2, label=r'$p_X^0 = \mathrm{Unif}[-1, 1]$')
axes[0, 2].fill_between(x3, p1_3, alpha=0.3, color=color_p1, step='mid')
axes[0, 2].plot(x3, p1_3, color=color_p1, lw=2, label=r'$p_X^1 = \mathrm{Unif}[0, 5]$')
axes[0, 2].set_title('(3) Uniform', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_ylim(0, 0.7)
axes[0, 2].legend(loc='upper right', fontsize=8)

# Row 2: Log density ratio plot
ax3_ratio = axes[1, 2]
# Overlap region [0, 1]: log ratio = log(2/5)
ax3_ratio.hlines(np.log(2/5), 0, 1, color=color_ratio, lw=3)
ax3_ratio.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Region [-1, 0): log ratio = -∞
ax3_ratio.annotate(r'$-\infty$', xy=(-0.5, -4), fontsize=10, ha='center', color='green', fontweight='bold')
ax3_ratio.annotate('', xy=(-0.5, -5.5), xytext=(-0.5, -3),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Region (1, 5]: log ratio = +∞
ax3_ratio.annotate(r'$+\infty$', xy=(3, 4), fontsize=10, ha='center', color='orange', fontweight='bold')
ax3_ratio.annotate('', xy=(3, 5.5), xytext=(3, 3),
                   arrowprops=dict(arrowstyle='->', color='orange', lw=2))

ax3_ratio.set_xlabel('x')
ax3_ratio.set_ylabel(r'$\log \frac{p_X^1}{p_X^0}$')
ax3_ratio.set_xlim(-2, 6)
ax3_ratio.set_ylim(-6, 6)
ax3_ratio.axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
ax3_ratio.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax3_ratio.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
ax3_ratio.axvline(x=5, color='gray', linestyle=':', alpha=0.5)

# ============== Case 4: Discrete point masses (disjoint support) ==============
np.random.seed(42)
x_points = np.array([-3, -1, 1, 3, 5])  # Support of P^0
y_points = np.array([-2, 0, 2, 4, 6])   # Support of P^1 (disjoint)
weights_0 = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
weights_1 = np.array([0.15, 0.25, 0.3, 0.2, 0.1])

# Row 1: Density plot
markerline0, stemlines0, baseline0 = axes[0, 3].stem(x_points, weights_0, basefmt=' ', 
                                                      linefmt='-', markerfmt='o')
plt.setp(stemlines0, color=color_p0, linewidth=2)
plt.setp(markerline0, color=color_p0, markersize=8)

markerline1, stemlines1, baseline1 = axes[0, 3].stem(y_points, weights_1, basefmt=' ',
                                                      linefmt='-', markerfmt='s')
plt.setp(stemlines1, color=color_p1, linewidth=2)
plt.setp(markerline1, color=color_p1, markersize=8)

axes[0, 3].set_title('(4) Discrete (Disjoint)', fontsize=12, fontweight='bold')
axes[0, 3].set_xlabel('x')
axes[0, 3].set_ylabel('Probability Mass')
axes[0, 3].set_ylim(0, 0.45)
axes[0, 3].set_xlim(-5, 8)

legend_elements = [Line2D([0], [0], marker='o', color=color_p0, lw=2, markersize=6,
                          label=r'$p_X^0 = \sum_i w_i^0 \delta_{x_i}$'),
                   Line2D([0], [0], marker='s', color=color_p1, lw=2, markersize=6,
                          label=r'$p_X^1 = \sum_j w_j^1 \delta_{y_j}$')]
axes[0, 3].legend(handles=legend_elements, loc='upper right', fontsize=8)

# Row 2: Density ratio - just warning message
ax4_ratio = axes[1, 3]
ax4_ratio.set_xlim(-5, 8)
ax4_ratio.set_ylim(0, 1)
ax4_ratio.set_xlabel('x')
ax4_ratio.set_ylabel(r'$\log \frac{p_X^1}{p_X^0}$')
ax4_ratio.set_xticks([])
ax4_ratio.set_yticks([])

ax4_ratio.text(0.5, 0.5, r'$\mathbf{Undefined}$' + '\n\nDisjoint supports:\n' + 
               r'$\mathrm{supp}(p_X^0) \cap \mathrm{supp}(p_X^1) = \emptyset$',
               ha='center', va='center', transform=ax4_ratio.transAxes,
               fontsize=11, color='red',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='red', lw=2))

# Add row labels on the left
# fig.text(0.02, 0.72, 'Densities', fontsize=14, fontweight='bold', rotation=90, va='center')
# fig.text(0.02, 0.28, r'Log Density Ratio $\log \frac{p_X^1}{p_X^0}$', fontsize=12, fontweight='bold', rotation=90, va='center')

plt.tight_layout()
plt.subplots_adjust(left=0.06)
plt.savefig('figures/density_ratio_plots.pdf', format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Plot saved successfully!")