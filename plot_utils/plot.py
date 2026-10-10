import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

rename_map = {
    # Baselines: all-caps, and did -> DiM
    "aipw": "AIPW",
    "sipw": "SIPW",
    "ipw": "IPW",
    "or": "OR",
    "did": "DiM",
    # Ours -> STE
    "global(bias)": r"STE($\eta_{gs}$)",
    "none": r"STE($\eta=0$)",
    "oracle": r"STE($\eta^{\ast}$)",
    "oracle (1st order)": r"STE($\eta^{\ast}$)",  # Handle both versions
    "ratio": r"STE($\hat{\eta}_{chisq}$)",
    "ipm(w1)": r"STE($\hat{\eta}_{w1}$)",
    "ipm(w2)": r"STE($\hat{\eta}_{w2}$)"
}

def rename_strategies(df, strategy_col='strategy'):
    """
    Rename strategies according to paper naming convention.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with strategy column
    strategy_col : str
        Name of the strategy column
        
    Returns:
    --------
    pd.DataFrame with renamed strategies
    """
    df_renamed = df.copy()
    df_renamed[strategy_col] = df_renamed[strategy_col].replace(rename_map)
    return df_renamed
    
def order_strategies(strategies):
    """
    Order strategies: baselines first, then ours methods.
    
    Parameters:
    -----------
    strategies : list or array
        List of strategy names
        
    Returns:
    --------
    list : Ordered strategy names
    """
    # Define the full display order: baselines first, then STE (ours) methods.
    baseline_order = ['AIPW', 'SIPW', 'IPW', 'OR', 'DiM']
    ours_order = [
        r"STE($\eta=0$)",
        r"STE($\eta^{\ast}$)",
        r"STE($\eta_{gs}$)",
        r"STE($\hat{\eta}_{w1}$)",
        r"STE($\hat{\eta}_{w2}$)",
        r"STE($\hat{\eta}_{chisq}$)",
    ]
    full_order = baseline_order + ours_order

    # Strategies present, sorted by the explicit order above
    ordered = [s for s in full_order if s in strategies]

    # Append any strategies not covered by the explicit order (alphabetical)
    leftovers = sorted(s for s in strategies if s not in full_order)

    return ordered + leftovers

def compute_statistics(df, group_cols=['$\\delta$', 'strategy']):
    r"""
    Compute bias, variance, and MSE statistics across seeds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: seed, $\delta$, strategy, mean_eta, bias, ATT_estimate, true_ATT
    group_cols : list
        Columns to group by
        
    Returns:
    --------
    pd.DataFrame with computed statistics
    """
    # Filter out errors
    df_clean = df[~df['has_error']].copy()
    
    # Compute variance as variance of ATT_estimate across seeds (bootstrap variance)
    # Compute bias statistics (mean and std across seeds)
    # Compute MSE = bias^2 + variance
    
    agg_dict = {
        'mean_eta': ['mean', 'std', 'count'],  # For methods with eta
        'bias': ['mean', 'std', 'count'],       # Bias varies across seeds
        'ATT_estimate': ['mean', 'std', 'var'], # For computing variance
        'true_ATT': 'first',                     # True value (same for all seeds at given delta)
    }
    
    stats = df_clean.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                     for col in stats.columns.values]
    
    # Rename for clarity
    stats = stats.rename(columns={
        'bias_mean': 'mean_bias',
        'bias_std': 'std_bias',
        'bias_count': 'n_seeds',
        'ATT_estimate_var': 'variance',  # This is the variance across seeds (bootstrap variance)
        'ATT_estimate_mean': 'mean_estimate',
        'ATT_estimate_std': 'std_estimate',
    })
    
    # Compute MSE = E[(estimate - true)^2] = bias^2 + variance
    stats['mse'] = stats['mean_bias']**2 + stats['variance']
    
    # Compute standard error for bias (for error bars)
    stats['se_bias'] = stats['std_bias'] / np.sqrt(stats['n_seeds'])
    
    # For comparison: compute RMSE
    stats['rmse'] = np.sqrt(stats['mse'])
    
    return stats


def plot_bias_variance_mse(stats, 
                           figsize=(18, 5),
                           use_log_scale=False,
                           use_abs_bias=True,
                           error_bars='se',  # 'se' for standard error, 'ci95' for 95% CI
                           save_path=None,
                           title_suffix=""):
    """
    Create publication-quality plots for bias, variance, and MSE.
    
    Parameters:
    -----------
    stats : pd.DataFrame
        Statistics dataframe from compute_statistics()
    figsize : tuple
        Figure size
    use_log_scale : bool or dict
        If True, use log scale for all y-axes. If dict, specify for each metric: 
        {'bias': False, 'variance': True, 'mse': True}
    use_abs_bias : bool
        If True, plot absolute value of bias
    error_bars : str
        'se' for standard error, 'ci95' for 95% confidence interval, 'none' for no error bars
    save_path : str
        Path to save figure
    title_suffix : str
        Additional text for plot title
    """
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Define consistent markers and colors for each strategy
    # Rename strategies first
    # stats = rename_strategies(stats)

    # Get ordered strategies
    strategies = order_strategies(stats['strategy'].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
    
    # Create color palette
    colors = sns.color_palette("husl", n_colors=len(strategies))
    style_dict = {strategy: {'marker': markers[i % len(markers)], 
                             'color': colors[i],
                             'markersize': 7,
                             'linewidth': 2}
                  for i, strategy in enumerate(strategies)}
    
    # Compute error bars
    if error_bars == 'se':
        stats['bias_error'] = stats['se_bias']
        error_label = 'SE'
    elif error_bars == 'ci95':
        stats['bias_error'] = 1.96 * stats['se_bias']
        error_label = '95% CI'
    else:
        stats['bias_error'] = 0
        error_label = None
    
    # Handle log scale settings
    if isinstance(use_log_scale, bool):
        log_settings = {'bias': use_log_scale, 'variance': use_log_scale, 'mse': use_log_scale}
    else:
        log_settings = use_log_scale
    
    # Sort by delta for proper line plotting
    stats = stats.sort_values('$\\delta$')
    
    # ------------------------
    # Plot 1: Bias
    # ------------------------
    ax = axes[0]
    for strategy in strategies:
        subset = stats[stats['strategy'] == strategy].copy()
        
        # Use absolute bias if requested
        y_vals = np.abs(subset['mean_bias']) if use_abs_bias else subset['mean_bias']
        y_err = subset['bias_error'] if error_bars != 'none' else None
        
        label = strategy
        # Special label handling
        if strategy == "oracle":
            label = "oracle (1st order)"
        
        ax.errorbar(
            subset['$\\delta$'],
            y_vals,
            yerr=y_err,
            label=label,
            marker=style_dict[strategy]['marker'],
            color=style_dict[strategy]['color'],
            markersize=style_dict[strategy]['markersize'],
            linewidth=style_dict[strategy]['linewidth'],
            capsize=4,
            capthick=1.5,
            alpha=0.85,
            linestyle='-',
        )
    
    ax.set_xlabel(r'$s$', fontsize=13, fontweight='bold')
    ylabel = r'$|\mathrm{Bias}|$' if use_abs_bias else 'Bias'
    if error_label:
        ylabel += f' (± {error_label})'
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if log_settings.get('bias', False):
        ax.set_yscale('log')
    ax.tick_params(labelsize=11)
    
    # ------------------------
    # Plot 2: Variance
    # ------------------------
    ax = axes[1]
    for strategy in strategies:
        subset = stats[stats['strategy'] == strategy].copy()
        
        label = strategy
        if strategy == "oracle":
            label = "oracle (1st order)"
        
        ax.plot(
            subset['$\\delta$'],
            subset['variance'],
            label=label,
            marker=style_dict[strategy]['marker'],
            color=style_dict[strategy]['color'],
            markersize=style_dict[strategy]['markersize'],
            linewidth=style_dict[strategy]['linewidth'],
            alpha=0.85,
            linestyle='-',
        )
    
    ax.set_xlabel(r'$s$', fontsize=13, fontweight='bold')
    ax.set_ylabel('Variance (Bootstrap)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if log_settings.get('variance', False):
        ax.set_yscale('log')
    ax.tick_params(labelsize=11)
    
    # ------------------------
    # Plot 3: MSE
    # ------------------------
    ax = axes[2]
    for strategy in strategies:
        subset = stats[stats['strategy'] == strategy].copy()
        
        label = strategy
        if strategy == "oracle":
            label = "oracle (1st order)"
        
        ax.plot(
            subset['$\\delta$'],
            subset['mse'],
            label=label,
            marker=style_dict[strategy]['marker'],
            color=style_dict[strategy]['color'],
            markersize=style_dict[strategy]['markersize'],
            linewidth=style_dict[strategy]['linewidth'],
            alpha=0.85,
            linestyle='-',
        )
    
    ax.set_xlabel(r'$s$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'MSE ($\mathrm{Bias}^2 + \mathrm{Variance}$)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if log_settings.get('mse', False):
        ax.set_yscale('log')
    ax.tick_params(labelsize=11)
    
    # Add legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=len(labels),
               frameon=True,
               fontsize=11,
               columnspacing=1.5)
    
    # # Add overall title if provided
    # if title_suffix:
    #     fig.suptitle(f'Bias-Variance-MSE Decomposition {title_suffix}', 
    #                 fontsize=14, fontweight='bold', y=1.08)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        # Always write a vector PDF regardless of the extension passed in
        save_path = str(Path(save_path).with_suffix('.pdf'))
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, axes

def plot_methods_with_eta(stats_with_eta,
                          figsize=(18, 5),
                          save_path=None,
                          one_minus_eta=False):
    """
    Create separate plots for methods with eta parameter to show eta values.

    Parameters:
    -----------
    stats_with_eta : pd.DataFrame
        Statistics for methods that have mean_eta values
    figsize : tuple
        Figure size
    save_path : str
        Path to save figure
    one_minus_eta : bool
        If True, plot 1 - eta (and label the axis accordingly) instead of eta.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get strategies present in this data
    strategies = order_strategies(stats_with_eta['strategy'].unique())
    
    # IMPORTANT: Create color palette for ALL possible strategies (including baselines)
    # to ensure consistent colors with the main plot
    all_strategies = order_strategies([
        'AIPW', 'SIPW', 'IPW', 'OR', 'DiM',  # baselines
        r"STE($\eta_{gs}$)", r"STE($\eta=0$)",
        r"STE($\eta^{\ast}$)", r"STE($\hat{\eta}_{chisq}$)",  # STE methods
        r"STE($\hat{\eta}_{w1}$)", r"STE($\hat{\eta}_{w2}$)"
    ])
    
    # Generate colors for ALL strategies
    all_colors = sns.color_palette("husl", n_colors=len(all_strategies))
    all_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
    
    # Create full style dictionary
    full_style_dict = {strategy: {'marker': all_markers[i % len(all_markers)], 
                                   'color': all_colors[i],
                                   'markersize': 8,
                                   'linewidth': 2.5}
                       for i, strategy in enumerate(all_strategies)}
    
    # Use only the styles for strategies present in this plot
    style_dict = {s: full_style_dict[s] for s in strategies if s in full_style_dict}
    
    stats_with_eta = stats_with_eta.sort_values('$\\delta$')
    
    # Plot 1: Mean eta
    ax = axes[0]
    for strategy in strategies:
        subset = stats_with_eta[stats_with_eta['strategy'] == strategy].copy()
        
        label = strategy
        
        # Show error bars for eta if available
        eta_err = subset.get('mean_eta_std', None)

        # Optionally plot 1 - eta (error magnitude is unchanged by the shift)
        eta_vals = 1 - subset['mean_eta_mean'] if one_minus_eta else subset['mean_eta_mean']

        ax.errorbar(
            subset['$\\delta$'],
            eta_vals,
            yerr=eta_err if eta_err is not None else None,
            label=label,
            marker=style_dict[strategy]['marker'],
            color=style_dict[strategy]['color'],
            markersize=style_dict[strategy]['markersize'],
            linewidth=style_dict[strategy]['linewidth'],
            capsize=4,
            capthick=1.5,
            alpha=0.85,
            linestyle='-',
        )
    
    ax.set_xlabel(r'$s$', fontsize=13, fontweight='bold')
    # Notation: the plotted (1 - eta) quantity is labeled \hat{\eta}
    eta_ylabel = r'$\hat{\eta}$' if one_minus_eta else r'$\bar{\eta}$ (Mean Weight)'
    ax.set_ylabel(eta_ylabel, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    # Plot 2: Bias for methods with eta
    ax = axes[1]
    for strategy in strategies:
        subset = stats_with_eta[stats_with_eta['strategy'] == strategy].copy()
        
        label = strategy
        
        ax.errorbar(
            subset['$\\delta$'],
            np.abs(subset['mean_bias']),
            yerr=subset['se_bias'],
            label=label,
            marker=style_dict[strategy]['marker'],
            color=style_dict[strategy]['color'],
            markersize=style_dict[strategy]['markersize'],
            linewidth=style_dict[strategy]['linewidth'],
            capsize=4,
            capthick=1.5,
            alpha=0.85,
            linestyle='-',
        )
    
    ax.set_xlabel(r'$s$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$|\mathrm{Bias}|$ (± SE)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Plot 3: MSE for methods with eta
    ax = axes[2]
    for strategy in strategies:
        subset = stats_with_eta[stats_with_eta['strategy'] == strategy].copy()
        
        label = strategy
        
        ax.plot(
            subset['$\\delta$'],
            subset['mse'],
            label=label,
            marker=style_dict[strategy]['marker'],
            color=style_dict[strategy]['color'],
            markersize=style_dict[strategy]['markersize'],
            linewidth=style_dict[strategy]['linewidth'],
            alpha=0.85,
            linestyle='-',
        )
    
    ax.set_xlabel(r'$s$', fontsize=13, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.02),
               ncol=len(labels),
               frameon=True,
               fontsize=11)
    
    # fig.suptitle(r'Methods with Tunable $\eta$ Parameter', 
    #             fontsize=14, fontweight='bold', y=1.08)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        # Always write a vector PDF regardless of the extension passed in
        save_path = str(Path(save_path).with_suffix('.pdf'))
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, axes
    


def generate_latex_table(stats, 
                         delta_values=None,
                         metrics=['mean_bias', 'variance', 'mse'],
                         caption="Bias, Variance, and MSE for different methods",
                         label="tab:bias_variance_mse",
                         format_precision=3,
                         save_path=None):
    """
    Generate a publication-ready LaTeX table.

    Parameters:
    -----------
    stats : pd.DataFrame
        Statistics dataframe
    delta_values : list
        Specific delta values to include (None for all)
    metrics : list
        Metrics to include in table
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    format_precision : int
        Number of decimal places
    save_path : str
        Path to save .tex file

    Returns:
    --------
    str : LaTeX table string
    """

    if delta_values is not None:
        stats = stats[stats['$\\delta$'].isin(delta_values)].copy()

    # Sort by delta and strategy
    stats = stats.sort_values(['$\\delta$', 'strategy'])

    # Create pivot tables for each metric
    tables = {}
    for metric in metrics:
        pivot = stats.pivot(index='strategy', columns='$\\delta$', values=metric)
        tables[metric] = pivot

    # Metric display names
    metric_names = {
        'mean_bias': 'Bias',
        'variance': 'Variance',
        'mse': 'MSE',
        'rmse': 'RMSE',
        'mean_eta_mean': r'$\bar{\eta}$',
    }

    # Start building LaTeX table
    n_cols = len(tables[metrics[0]].columns)
    col_format = 'l' + 'r' * n_cols

    latex_str = f"""% LaTeX table generated automatically
                \\begin{{table}}[htbp]
                    \\centering
                    \\caption{{{caption}}}
                    \\label{{{label}}}
                    \\resizebox{{\\textwidth}}{{!}}{{%
                """

    for i, metric in enumerate(metrics):
        pivot = tables[metric]

        # Add section title
        if i > 0:
            latex_str += "\n        \\midrule\n        \\midrule\n"

        header_cols = ' & '.join([f'$s={v:.1f}$' for v in pivot.columns])

        latex_str += f"""        \\begin{{tabular}}{{{col_format}}}
                \\toprule
                \\multicolumn{{{n_cols+1}}}{{c}}{{\\textbf{{{metric_names.get(metric, metric)}}}}} \\\\
                \\midrule
                \\textbf{{Method}} & {header_cols} \\\\
                \\midrule
        """

        # Add data rows
        for strategy in pivot.index:
            row_data = []
            for val in pivot.loc[strategy]:
                if pd.isna(val):
                    row_data.append('---')
                else:
                    # Format with scientific notation if very large/small
                    if abs(val) < 0.001 or abs(val) > 10000:
                        row_data.append(f'{val:.{format_precision}e}')
                    else:
                        row_data.append(f'{val:.{format_precision}f}')

            # Clean up strategy name for LaTeX
            strategy_name = strategy.replace('_', '\\_')
            latex_str += f"            {strategy_name} & {' & '.join(row_data)} \\\\\n"

        latex_str += "            \\bottomrule\n"
        latex_str += "        \\end{tabular}"

        if i < len(metrics) - 1:
            latex_str += "\n        \\vspace{0.5cm}\n\n"

    latex_str += """
                    }
                \\end{table}
                """

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"✓ LaTeX table saved to: {save_path}")

    return latex_str


def generate_wide_latex_table(stats,
                              delta_values=None,
                              caption="Performance comparison across methods and tuning parameters",
                              label="tab:performance_wide",
                              format_precision=2,
                              save_path=None):
    """
    Generate a wide-format LaTeX table with all metrics in one table.
    Each row is a (strategy, delta) pair, columns are metrics.
    
    Parameters:
    -----------
    stats : pd.DataFrame
        Statistics dataframe
    delta_values : list
        Specific delta values to include
    caption : str
        Table caption
    label : str
        LaTeX label
    format_precision : int
        Decimal places
    save_path : str
        Path to save .tex file
    """
    
    if delta_values is not None:
        stats = stats[stats['$\\delta$'].isin(delta_values)].copy()
    
    stats = stats.sort_values(['strategy', '$\\delta$'])
    
    # Select columns to display
    display_cols = {
        '$\\delta$': r'$\delta$',
        'mean_eta_mean': r'$\bar{\eta}$',
        'mean_bias': 'Bias',
        'se_bias': 'SE(Bias)',
        'variance': 'Variance',
        'mse': 'MSE',
    }
    
    # Build LaTeX
    col_spec = 'l' + ('r' * (len(display_cols) - 1))
    header_cols = ' & '.join([f'\\textbf{{{v}}}' for k, v in display_cols.items() if k != 'strategy'])
    
    latex_str = f"""% Wide-format LaTeX table
				\\begin{{table}}[htbp]
				    \\centering
				    \\caption{{{caption}}}
				    \\label{{{label}}}
				    \\scriptsize
				    \\begin{{tabular}}{{{col_spec}}}
				        \\toprule
				        \\textbf{{Method}} & {header_cols} \\\\
				        \\midrule
				"""
    
    current_strategy = None
    for idx, row in stats.iterrows():
        strategy = row['strategy']
        
        # Add separator between strategies
        if current_strategy is not None and strategy != current_strategy:
            latex_str += "        \\midrule\n"
        current_strategy = strategy
        
        # Format row
        row_values = []
        for col in display_cols.keys():
            if col == '$\\delta$':
                continue  # Skip, handled separately
            
            val = row.get(col, np.nan)
            if pd.isna(val) or (col == 'mean_eta_mean' and val == 0 and pd.isna(row.get('mean_eta_std', np.nan))):
                row_values.append('---')
            else:
                if abs(val) < 0.001 or abs(val) > 10000:
                    row_values.append(f'{val:.{format_precision}e}')
                else:
                    row_values.append(f'{val:.{format_precision}f}')
        
        delta_val = row['$\\delta$']
        strategy_display = strategy.replace('_', '\\_')
        
        latex_str += f"        {strategy_display} & {delta_val:.1f} & {' & '.join(row_values)} \\\\\n"
    
    latex_str += """        \\bottomrule
				    \\end{tabular}
				\\end{table}
				"""
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"✓ Wide LaTeX table saved to: {save_path}")
    
    return latex_str


def generate_compact_latex_table(stats,
                                 selected_deltas=[0, 2, 5, 10],
                                 top_k_methods=5,
                                 metric='mse',
                                 caption="Top performing methods by MSE",
                                 label="tab:top_methods",
                                 save_path=None):
    """
    Generate a compact table showing only top-k methods at selected delta values.
    
    Parameters:
    -----------
    stats : pd.DataFrame
        Statistics dataframe
    selected_deltas : list
        Delta values to show
    top_k_methods : int
        Number of top methods to show (by metric at each delta)
    metric : str
        Metric to rank by ('mse', 'mean_bias', etc.)
    caption : str
        Table caption
    label : str
        LaTeX label
    save_path : str
        Path to save
    """
    
    # Filter selected deltas
    stats_subset = stats[stats['$\\delta$'].isin(selected_deltas)].copy()
    
    # For each delta, get top k methods
    top_methods = []
    for delta in selected_deltas:
        delta_data = stats_subset[stats_subset['$\\delta$'] == delta].copy()
        
        # Rank by absolute value if bias, otherwise by value
        if 'bias' in metric.lower():
            delta_data['rank_metric'] = delta_data[metric].abs()
        else:
            delta_data['rank_metric'] = delta_data[metric]
        
        top_k = delta_data.nsmallest(top_k_methods, 'rank_metric')
        top_k['rank'] = range(1, len(top_k) + 1)
        top_methods.append(top_k)
    
    # Combine
    top_df = pd.concat(top_methods, ignore_index=True)
    
    # Build table
    latex_str = f"""% Compact table showing top methods
				\\begin{{table}}[htbp]
				    \\centering
				    \\caption{{{caption}}}
				    \\label{{{label}}}
				    \\begin{{tabular}}{{clrrr}}
				        \\toprule
				        $s$ & \\textbf{{Rank}} & \\textbf{{Method}} & \\textbf{{Bias}} & \\textbf{{MSE}} \\\\
				        \\midrule
				"""
    
    for delta in selected_deltas:
        delta_top = top_df[top_df['$\\delta$'] == delta].sort_values('rank')
        
        for idx, row in delta_top.iterrows():
            delta_str = f"{delta:.1f}" if row['rank'] == 1 else ""
            rank = int(row['rank'])
            method = row['strategy'].replace('_', '\\_')
            bias_val = f"{row['mean_bias']:.3f}"
            mse_val = f"{row['mse']:.3f}"
            
            latex_str += f"        {delta_str} & {rank} & {method} & {bias_val} & {mse_val} \\\\\n"
        
        if delta != selected_deltas[-1]:
            latex_str += "        \\midrule\n"
    
    latex_str += """        \\bottomrule
				    \\end{tabular}
				\\end{table}
				"""
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"✓ Compact table saved to: {save_path}")
    
    return latex_str



def generate_latex_table_horizontal(stats, 
                         delta_value=None,
                         metrics=['mean_bias', 'variance', 'mse'],
                         caption="Bias, Variance, and MSE for different methods",
                         label="tab:bias_variance_mse",
                         format_precision=3,
                         save_path=None):
    """
    Generate a publication-ready LaTeX table with methods as columns and metrics as rows.
    
    Parameters:
    -----------
    stats : pd.DataFrame
        Statistics dataframe with columns: strategy, $\\delta$, and metric columns
    delta_value : float
        Specific delta value to include (if None, uses all data or first delta)
    metrics : list
        Metrics to include as rows in table
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    format_precision : int
        Number of decimal places
    save_path : str
        Path to save .tex file
        
    Returns:
    --------
    str : LaTeX table string
    """
    # Filter to specific delta if provided
    if delta_value is not None:
        stats = stats[stats['$\\delta$'] == delta_value].copy()
    
    # Sort by strategy
    # stats = stats.sort_values('strategy').reset_index(drop=True)
    strategies = order_strategies(stats['strategy'].unique())
    
    # Get list of strategies (methods)
    # strategies = stats['strategy'].tolist()
    n_methods = len(strategies)
    
    # Metric display names
    metric_names = {
        'mean_bias': 'Bias',
        'variance': 'Variance',
        'mse': 'MSE',
        'rmse': 'RMSE',
        'mean_eta_mean': r'$\bar{\eta}$',
    }
    
    # Column format: metric name + one column per method
    col_format = 'l' + 'r' * n_methods
    
    # Build header with method names
    header_cols = ' & '.join([format_strategy_name(s) for s in strategies])
    
    # Start building LaTeX table
    latex_str = f"""\\begin{{table}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\resizebox{{\\textwidth}}{{!}}{{%
        \\begin{{tabular}}{{{col_format}}}
            \\toprule
            \\textbf{{Metric}} & {header_cols} \\\\
            \\midrule
        """
    
    # Add data rows (one per metric)
    for metric in metrics:
        metric_display = metric_names.get(metric, metric)

        # Compute a shared scientific-notation exponent across methods for this metric.
        # We scale by 10^exp where exp = floor(log10(max_abs_value)).
        metric_vals = []
        for strategy in strategies:
            v = stats.loc[stats['strategy'] == strategy, metric].values[0]
            if v is None or pd.isna(v):
                continue
            metric_vals.append(float(v))

        if len(metric_vals) == 0:
            exp = 0
            scale = 1.0
        else:
            abs_vals = np.abs(np.asarray(metric_vals, dtype=float))
            # Use the smallest *non-zero* magnitude to choose a common exponent.
            # This avoids exp=-inf when some methods have exactly 0 for a metric.
            nonzero_abs = abs_vals[abs_vals > 0]
            if nonzero_abs.size == 0:
                exp = 0
                scale = 1.0
            else:
                min_abs = float(np.min(nonzero_abs))
                exp = int(np.floor(np.log10(min_abs)))
                scale = float(10 ** exp)

        metric_label = metric_display
        if exp != 0:
            metric_label = f"{metric_display} $(\\times 10^{{{exp}}})$"

        # Get values for this metric across all methods, scaled consistently
        row_data = []
        for strategy in strategies:
            val = stats.loc[stats['strategy'] == strategy, metric].values[0]

            if pd.isna(val) or val is None:
                row_data.append('---')
            else:
                scaled = float(val) / scale
                # Keep negatives in math mode for LaTeX consistency
                if scaled < 0:
                    row_data.append(f"${scaled:.{format_precision}f}$")
                else:
                    row_data.append(f"{scaled:.{format_precision}f}")

        latex_str += f"    \\textbf{{{metric_label}}} & {' & '.join(row_data)} \\\\\n"
    
    latex_str += """    \\bottomrule
        \\end{tabular}%
        }
        \\end{table}
        """
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"✓ LaTeX table saved to: {save_path}")
    
    return latex_str


def format_strategy_name(strategy):
    """Format strategy names for LaTeX display."""
    # Handle 'STE' (ours) variants
    if 'STE' in strategy or 'ours' in strategy:
        if 'eta=0' in strategy or 'eta=1' in strategy or 'eta_1' in strategy:
            return r"STE($\eta=0$)"
        elif 'eta_gs' in strategy or 'gs' in strategy:
            return r"STE($\eta_{gs}^{\ast}$)"
        elif 'chisq' in strategy:
            return r"STE($\hat{\eta}_{\text{chisq}}$)"
        elif 'w1' in strategy:
            return r"STE($\hat{\eta}_{w1}$)"
        elif 'w2' in strategy:
            return r"STE($\hat{\eta}_{w2}$)"
        else:
            return "STE"

    # Standard methods - just escape underscores
    return strategy.replace('_', r'\_')


def format_number(val, precision):
    """Format number with appropriate notation and handle negatives."""
    # Handle negative numbers - wrap in math mode
    if val < 0:
        if abs(val) < 0.001 or abs(val) > 10000:
            return f"${val:.{precision}e}$"
        else:
            return f"${val:.{precision}f}$"
    else:
        if abs(val) < 0.001 or abs(val) > 10000:
            return f"{val:.{precision}e}"
        else:
            return f"{val:.{precision}f}"

# ============================================================
# Main execution example
# ============================================================

if __name__ == "__main__":
    # d = 5
    # max_avg_weight = 1.1
    # outcome_model_type='linear'
    # outcome_type='quadratic'
    # dist = 'norm'

    # # Load data
    # csv_path = f"figs/{dist}_shift/{d}d_{dist}_{outcome_type}_{outcome_model_type}.csv"
    # output_dir = f"figs/{dist}_shift/"

    # df = pd.read_csv(csv_path)
    # # filter out the convex hull projection method
    # # Keep all rows where strategy is NOT 'individual'
    # df = df[df['strategy'] != 'individual']

    # # Rename strategies before computing statistics
    # df = rename_strategies(df)
    # print(df.columns)
    # df['mean_eta'] = 1-df['mean_eta'].values


    # print("="*70)
    # print("BIAS-VARIANCE-MSE ANALYSIS")
    # print("="*70)

    # # Compute statistics
    # print("\n[1/6] Computing statistics across seeds...")
    # stats = compute_statistics(df)
    # print(f"✓ Statistics computed for {len(stats)} (strategy, delta) combinations")


    # # -------------------------
    # # Plot 1: All methods comparison (standard view)
    # # -------------------------
    # print("\n[2/6] Creating main comparison plot...")
    # fig1, _ = plot_bias_variance_mse(
    #     stats,
    #     figsize=(18, 5),
    #     use_log_scale=False,
    #     use_abs_bias=True,
    #     error_bars='se',
    #     save_path=output_dir + f"{d}d_{dist}_{outcome_type}_{outcome_model_type}.png",
    #     title_suffix=""
    # )

    # # -------------------------
    # # Plot 2: All methods with log scale
    # # -------------------------
    # print("\n[3/6] Creating comparison plot with log scale...")
    # fig2, _ = plot_bias_variance_mse(
    #     stats,
    #     figsize=(18, 5),
    #     use_log_scale={'bias': False, 'variance': True, 'mse': True},
    #     use_abs_bias=True,
    #     error_bars='se',
    #     save_path=output_dir + f"{d}d_{dist}_{outcome_type}_{outcome_model_type}_log.png",
    #     title_suffix="(Log Scale)"
    # )

    # # -------------------------
    # # Plot 3: Methods with eta parameter
    # # -------------------------
    # print("\n[4/6] Creating plot for methods with eta parameter...")
    # # Filter methods that have eta
    # # strategies_with_eta = list(rename_map.values())
    # strategies_with_eta = [r"ours($\eta_{gs}$)", r"ours($\hat{\eta}_{chisq}$)",  # ours methods
    #     r"ours($\hat{\eta}_{w1}$)", r"ours($\hat{\eta}_{w2}$)"]
    # stats_eta = stats[stats['strategy'].isin(strategies_with_eta)].copy()
    # print("eta strategies ", stats_eta.strategy.unique())

    # fig3, _ = plot_methods_with_eta(
    #     stats_eta,
    #     figsize=(18, 5),
    #     save_path=output_dir + f"{d}d_{dist}_{outcome_type}_{outcome_model_type}_etas.png"
    # )

    # # -------------------------
    # # Generate LaTeX tables
    # # -------------------------
    # print("\n[5/6] Generating LaTeX tables...")

    # # Table 1: Standard format (separate sections for each metric)
    # table1 = generate_latex_table(
    #     stats,
    #     delta_values=[0, 2, 4, 6, 8, 10],
    #     metrics=['mean_bias', 'variance', 'mse'],
    #     caption="Bias, Variance, and MSE decomposition for different estimation methods across tuning parameter values",
    #     label="tab:bias_variance_mse",
    #     format_precision=3,
    #     save_path=output_dir + "table_bias_variance_mse.tex"
    # )

    # # Table 2: Wide format (all metrics in one table)
    # table2 = generate_wide_latex_table(
    #     stats,
    #     delta_values=[0, 2, 4, 6, 8, 10],
    #     caption="Comprehensive performance metrics for all estimation methods",
    #     label="tab:performance_wide",
    #     format_precision=2,
    #     save_path=output_dir + "table_performance_wide.tex"
    # )

    # # Table 3: Compact format (top methods only)
    # table3 = generate_compact_latex_table(
    #     stats,
    #     selected_deltas=[0, 2, 5, 10],
    #     top_k_methods=5,
    #     metric='mse',
    #     caption="Top 5 methods by MSE at selected tuning parameter values",
    #     label="tab:top_methods_mse",
    #     save_path=output_dir + "table_top_methods.tex"
    # )

    # # -------------------------
    # # Generate summary statistics CSV
    # # -------------------------
    # print("\n[6/6] Saving summary statistics...")
    # stats.to_csv(output_dir + "summary_statistics.csv", index=False)
    # print(f"✓ Summary statistics saved")

    # print("\n" + "="*70)
    # print("ANALYSIS COMPLETE!")
    # print("="*70)
    # print(f"\nAll outputs saved to: {output_dir}")
    # print("\nGenerated files:")
    # print("  Figures:")
    # print("    - bias_variance_mse_comparison.png")
    # print("    - bias_variance_mse_comparison_log.png")
    # print("    - methods_with_eta_comparison.png")
    # print("  Tables:")
    # print("    - table_bias_variance_mse.tex")
    # print("    - table_performance_wide.tex")
    # print("    - table_top_methods.tex")
    # print("  Data:")
    # print("    - summary_statistics.csv")
    # print("\n" + "="*70)


    # d = 20
    # max_avg_weight = 1.1
    dist = 'rhc'
    # outcome_model_type='quadratic'
    # dist = 'lalonde'
    outcome_model_type='quadratic'
    # outcome_type='quadratic'

    # dist = "meps"
    # outcome_model_type='quadratic'
    

    # Load data
    csv_path = f"results/{dist}/{dist}_{outcome_model_type}.csv"
    output_dir = f"figs/{dist}/"

    df = pd.read_csv(csv_path)
    # filter out the convex hull projection method
    # Keep all rows where strategy is NOT 'individual'
    df = df[df['strategy'] != 'individual']

    # Rename strategies before computing statistics
    df = rename_strategies(df)
    print(df.columns)
    df['mean_eta'] = 1-df['mean_eta'].values


    print("="*70)
    print("BIAS-VARIANCE-MSE ANALYSIS")
    print("="*70)

    # Compute statistics
    print("\n[1/6] Computing statistics across seeds...")
    stats = compute_statistics(df)
    print(f"✓ Statistics computed for {len(stats)} (strategy, delta) combinations")

    table = generate_latex_table_horizontal(
        stats,
        delta_value=0,
        metrics=['mean_bias', 'variance', 'mse'],
        caption="Bias, Variance, and MSE decomposition for different estimation methods across tuning parameter values",
        label="tab:bias_variance_mse",
        format_precision=3,
        save_path=output_dir + f"table_bias_variance_mse_{outcome_model_type}.tex"
    )


