import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import json

from .gen_data import standardize_data


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

def visualize_lalonde_overlap(
    data: Dict,
    save_dir: str = "figures",
    version: str = "psid",
    show_plots: bool = True
):
    """
    Comprehensive overlap visualization for LaLonde dataset.
    
    Parameters:
    -----------
    data : dict
        Dictionary from load_lalonde_data() containing X_control, X_treatment, etc.
    save_dir : str
        Directory to save figures
    version : str
        Dataset version name for plot titles
    show_plots : bool
        Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    X_control = data['X_control']
    X_treatment = data['X_treatment']
    Y_control = data['Y_control']
    Y_treatment = data['Y_treatment']
    covariate_names = data['covariate_names']
    
    # Combine data for visualization
    X_all = np.vstack([X_control, X_treatment])
    T_all = np.concatenate([np.zeros(len(X_control)), np.ones(len(X_treatment))])
    Y_all = np.concatenate([Y_control, Y_treatment])
    
    # Create figure with multiple subplots
    print(f"Creating overlap visualizations for {version} dataset...")
    
    # 1. Covariate distributions
    plot_covariate_distributions(X_control, X_treatment, covariate_names, 
                                 version, save_dir, show_plots)
    
    # 2. Propensity score overlap
    plot_propensity_overlap(X_all, T_all, covariate_names, 
                           version, save_dir, show_plots)
    
    # 3. Standardized mean differences (Love plot)
    plot_standardized_differences(X_control, X_treatment, covariate_names,
                                 version, save_dir, show_plots)
    
    # 4. 2D projections (PCA)
    plot_dimensionality_reduction(X_control, X_treatment, Y_control, Y_treatment,
                                 version, save_dir, show_plots)
    
    # 5. Covariate balance table
    create_balance_table(X_control, X_treatment, covariate_names, version, save_dir)
    
    # 6. Kernel density overlap
    plot_kernel_density_overlap(X_control, X_treatment, covariate_names,
                               version, save_dir, show_plots)
    
    print(f"\n✓ All visualizations saved to {save_dir}/")

def plot_covariate_distributions(X_control, X_treatment, covariate_names, 
                                 version, save_dir, show_plots):
    """Plot distributions of each covariate for treated vs control."""
    n_covariates = len(covariate_names)
    n_cols = 2
    n_rows = (n_covariates + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten()
    
    for i, name in enumerate(covariate_names):
        ax = axes[i]
        
        # Histogram + KDE
        ax.hist(X_control[:, i], bins=30, alpha=0.5, density=True, 
               label='Control', color='blue', edgecolor='black')
        ax.hist(X_treatment[:, i], bins=30, alpha=0.5, density=True,
               label='Treated', color='red', edgecolor='black')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        try:
            kde_control = gaussian_kde(X_control[:, i])
            kde_treatment = gaussian_kde(X_treatment[:, i])
            x_range = np.linspace(X_control[:, i].min(), 
                                 X_control[:, i].max(), 100)
            ax.plot(x_range, kde_control(x_range), 'b-', linewidth=2, alpha=0.7)
            ax.plot(x_range, kde_treatment(x_range), 'r-', linewidth=2, alpha=0.7)
        except:
            pass
        
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add vertical lines for means
        ax.axvline(X_control[:, i].mean(), color='blue', 
                  linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(X_treatment[:, i].mean(), color='red',
                  linestyle='--', linewidth=2, alpha=0.7)
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Covariate Distributions: {version.upper()}', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/covariate_distributions_{version}.png', 
                dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def plot_propensity_overlap(X_all, T_all, covariate_names, version, save_dir, show_plots):
    """Plot propensity score distribution and overlap."""
    # Fit logistic regression for propensity scores
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_all, T_all)
    propensity_scores = lr.predict_proba(X_all)[:, 1]
    
    ps_control = propensity_scores[T_all == 0]
    ps_treatment = propensity_scores[T_all == 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overlapping histograms
    ax = axes[0, 0]
    ax.hist(ps_control, bins=50, alpha=0.6, density=True, 
           label='Control', color='blue', edgecolor='black')
    ax.hist(ps_treatment, bins=50, alpha=0.6, density=True,
           label='Treated', color='red', edgecolor='black')
    ax.set_xlabel('Propensity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Propensity Score Distributions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add statistics
    overlap_pct = calculate_overlap_percentage(ps_control, ps_treatment)
    ax.text(0.05, 0.95, f'Overlap: {overlap_pct:.1f}%', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    # 2. Mirrored histogram
    ax = axes[0, 1]
    ax.hist(ps_control, bins=50, alpha=0.7, density=True,
           label='Control', color='blue', edgecolor='black')
    ax.hist(ps_treatment, bins=50, alpha=0.7, density=True,
           label='Treated', color='red', edgecolor='black', 
           weights=-np.ones_like(ps_treatment))
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Propensity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Mirrored Propensity Score', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # 3. Box plots
    ax = axes[1, 0]
    positions = [1, 2]
    bp = ax.boxplot([ps_control, ps_treatment], positions=positions,
                     widths=0.6, patch_artist=True,
                     labels=['Control', 'Treated'])
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    ax.set_ylabel('Propensity Score', fontsize=12)
    ax.set_title('Propensity Score Box Plots', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Jittered scatter
    ax = axes[1, 1]
    np.random.seed(42)
    jitter_control = np.random.normal(0, 0.05, len(ps_control))
    jitter_treatment = np.random.normal(1, 0.05, len(ps_treatment))
    
    ax.scatter(jitter_control, ps_control, alpha=0.3, s=20, 
              color='blue', label='Control')
    ax.scatter(jitter_treatment, ps_treatment, alpha=0.3, s=20,
              color='red', label='Treated')
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Treated'])
    ax.set_ylabel('Propensity Score', fontsize=12)
    ax.set_title('Propensity Score Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'Propensity Score Overlap Analysis: {version.upper()}',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/propensity_overlap_{version}.png', 
                dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"PROPENSITY SCORE STATISTICS ({version.upper()})")
    print(f"{'='*60}")
    print(f"Control:   mean={ps_control.mean():.3f}, std={ps_control.std():.3f}")
    print(f"           min={ps_control.min():.3f}, max={ps_control.max():.3f}")
    print(f"Treated:   mean={ps_treatment.mean():.3f}, std={ps_treatment.std():.3f}")
    print(f"           min={ps_treatment.min():.3f}, max={ps_treatment.max():.3f}")
    print(f"Overlap:   {overlap_pct:.1f}%")
    print(f"{'='*60}\n")


def plot_standardized_differences(X_control, X_treatment, covariate_names,
                                  version, save_dir, show_plots):
    """Love plot showing standardized mean differences."""
    smd = []
    for i in range(X_control.shape[1]):
        mean_diff = X_treatment[:, i].mean() - X_control[:, i].mean()
        pooled_std = np.sqrt((X_control[:, i].std()**2 + 
                             X_treatment[:, i].std()**2) / 2)
        smd.append(mean_diff / pooled_std if pooled_std > 0 else 0)
    
    smd = np.array(smd)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if abs(s) > 0.1 else 'orange' if abs(s) > 0.05 else 'green' 
             for s in smd]
    
    y_pos = np.arange(len(covariate_names))
    ax.barh(y_pos, smd, color=colors, edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariate_names, fontsize=11)
    ax.set_xlabel('Standardized Mean Difference', fontsize=12)
    ax.set_title(f'Covariate Balance (Love Plot): {version.upper()}', 
                fontsize=14, fontweight='bold')
    
    # Add reference lines
    ax.axvline(0, color='black', linewidth=1.5)
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='|SMD| = 0.1')
    ax.axvline(-0.1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(0.05, color='orange', linestyle='--', alpha=0.5, label='|SMD| = 0.05')
    ax.axvline(-0.05, color='orange', linestyle='--', alpha=0.5)
    
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='x')
    
    # Add text annotation
    max_smd = np.abs(smd).max()
    balance_quality = "Excellent" if max_smd < 0.05 else "Good" if max_smd < 0.1 else "Poor"
    ax.text(0.02, 0.98, f'Max |SMD|: {max_smd:.3f}\nBalance: {balance_quality}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/love_plot_{version}.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"STANDARDIZED MEAN DIFFERENCES ({version.upper()})")
    print(f"{'='*60}")
    for name, s in zip(covariate_names, smd):
        status = "✓" if abs(s) < 0.05 else "!" if abs(s) < 0.1 else "✗"
        print(f"{status} {name:15s}: {s:7.3f}")
    print(f"{'='*60}\n")


def plot_dimensionality_reduction(X_control, X_treatment, Y_control, Y_treatment,
                                  version, save_dir, show_plots):
    """Plot PCA and t-SNE projections."""
    X_all = np.vstack([X_control, X_treatment])
    T_all = np.concatenate([np.zeros(len(X_control)), np.ones(len(X_treatment))])
    Y_all = np.concatenate([Y_control, Y_treatment])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)
    
    ax = axes[0]
    scatter_c = ax.scatter(X_pca[T_all==0, 0], X_pca[T_all==0, 1], 
                          c=Y_control, cmap='Blues', alpha=0.6, s=50,
                          edgecolors='black', linewidth=0.5, label='Control')
    scatter_t = ax.scatter(X_pca[T_all==1, 0], X_pca[T_all==1, 1],
                          c=Y_treatment, cmap='Reds', alpha=0.6, s=50,
                          edgecolors='black', linewidth=0.5, label='Treated')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA Projection', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter_c, ax=ax)
    cbar.set_label('Earnings (Control)', fontsize=10)
    
    # t-SNE (if dataset not too large)
    if len(X_all) < 5000:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_all)
        
        ax = axes[1]
        ax.scatter(X_tsne[T_all==0, 0], X_tsne[T_all==0, 1],
                  alpha=0.6, s=50, color='blue', edgecolors='black',
                  linewidth=0.5, label='Control')
        ax.scatter(X_tsne[T_all==1, 0], X_tsne[T_all==1, 1],
                  alpha=0.6, s=50, color='red', edgecolors='black',
                  linewidth=0.5, label='Treated')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('t-SNE Projection', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Dataset too large for t-SNE\n(>5000 samples)',
                    ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
        axes[1].axis('off')
    
    plt.suptitle(f'Dimensionality Reduction: {version.upper()}',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dimensionality_reduction_{version}.png',
                dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def plot_kernel_density_overlap(X_control, X_treatment, covariate_names,
                                version, save_dir, show_plots):
    """Plot kernel density estimates with shaded overlap regions."""
    n_covariates = len(covariate_names)
    n_cols = 2
    n_rows = (n_covariates + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows))
    axes = axes.flatten()
    
    from scipy.stats import gaussian_kde
    
    for i, name in enumerate(covariate_names):
        ax = axes[i]
        
        try:
            # Create KDEs
            kde_control = gaussian_kde(X_control[:, i])
            kde_treatment = gaussian_kde(X_treatment[:, i])
            
            # Create x-axis range
            x_min = min(X_control[:, i].min(), X_treatment[:, i].min())
            x_max = max(X_control[:, i].max(), X_treatment[:, i].max())
            x_range = np.linspace(x_min, x_max, 300)
            
            # Evaluate KDEs
            density_control = kde_control(x_range)
            density_treatment = kde_treatment(x_range)
            
            # Plot KDEs
            ax.plot(x_range, density_control, 'b-', linewidth=2.5, 
                   label='Control', alpha=0.8)
            ax.plot(x_range, density_treatment, 'r-', linewidth=2.5,
                   label='Treated', alpha=0.8)
            
            # Shade overlap region
            overlap = np.minimum(density_control, density_treatment)
            ax.fill_between(x_range, overlap, alpha=0.3, color='purple',
                           label='Overlap')
            
            # Calculate overlap coefficient
            overlap_coef = np.trapz(overlap, x_range) / \
                          np.trapz(np.maximum(density_control, density_treatment), x_range)
            
            ax.set_xlabel(name, fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{name} (Overlap: {overlap_coef:.2%})', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            
        except:
            ax.text(0.5, 0.5, f'Unable to compute KDE\nfor {name}',
                   ha='center', va='center', transform=ax.transAxes)
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Kernel Density Overlap: {version.upper()}',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/kernel_density_overlap_{version}.png',
                dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def create_balance_table(X_control, X_treatment, covariate_names, version, save_dir):
    """Create a detailed balance table."""
    results = []
    
    for i, name in enumerate(covariate_names):
        # Means
        mean_c = X_control[:, i].mean()
        mean_t = X_treatment[:, i].mean()
        
        # Standard deviations
        std_c = X_control[:, i].std()
        std_t = X_treatment[:, i].std()
        
        # Standardized mean difference
        pooled_std = np.sqrt((std_c**2 + std_t**2) / 2)
        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0
        
        # Variance ratio
        var_ratio = std_t**2 / std_c**2 if std_c > 0 else np.nan
        
        # t-test
        t_stat, p_value = stats.ttest_ind(X_treatment[:, i], X_control[:, i])
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(X_treatment[:, i], X_control[:, i])
        
        results.append({
            'Covariate': name,
            'Control Mean': mean_c,
            'Control SD': std_c,
            'Treated Mean': mean_t,
            'Treated SD': std_t,
            'SMD': smd,
            'Var Ratio': var_ratio,
            't-stat': t_stat,
            'p-value': p_value,
            'KS-stat': ks_stat,
            'KS p-value': ks_pvalue
        })
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(f'{save_dir}/balance_table_{version}.csv', index=False)
    
    # Pretty print
    print(f"\n{'='*100}")
    print(f"COVARIATE BALANCE TABLE ({version.upper()})")
    print(f"{'='*100}")
    print(df.to_string(index=False, float_format='%.3f'))
    print(f"{'='*100}\n")
    
    return df


def calculate_overlap_percentage(ps_control, ps_treatment):
    """Calculate percentage of overlap in propensity score distributions."""
    from scipy.stats import gaussian_kde
    
    # Create KDEs
    kde_c = gaussian_kde(ps_control)
    kde_t = gaussian_kde(ps_treatment)
    
    # Evaluate on common grid
    x_range = np.linspace(0, 1, 1000)
    density_c = kde_c(x_range)
    density_t = kde_t(x_range)
    
    # Calculate overlap coefficient
    overlap = np.minimum(density_c, density_t)
    overlap_pct = 100 * np.trapz(overlap, x_range) / \
                  np.trapz(np.maximum(density_c, density_t), x_range)
    
    return overlap_pct



def load_lalonde_data(
    path="data/lalond_experimental.csv", 
    download=False, 
    version='experimental', 
    include_categorical=False,
    compute_true_ATT=True,
    standardize=True,
    standardize_method: str = 'zscore',
    ):
    """
    Load LaLonde dataset and prepare for causal inference.
    
    Parameters:
    -----------
    version : str
        'experimental': NSW experimental data (RCT, true ATT known)
        'psid': NSW treated + PSID controls (observational, severe overlap issues)
        'cps': NSW treated + CPS controls (observational, moderate overlap issues)
    include_categorical : bool
        If True, include binary covariates. If False, only continuous covariates.
    standardize : bool
        Whether to standardize covariates
    standardize_method : str
        'zscore', 'minmax', or 'robust'
    Returns:
    --------
    Dictionary with X_control, X_treatment, Y_control, Y_treatment, 
    true_ATT (if experimental), outcome_info, covariate_names, n0, n1, d
    """
    
    if download:
        # Load data from Dehejia-Wahba's website
        base_url = "https://users.nber.org/~rdehejia/data/nswre74"
        
        if version == 'experimental':
            # Load experimental data (treated + experimental controls)
            treated_url = f"{base_url}_treated.txt"
            control_url = f"{base_url}_control.txt"
        elif version == 'psid':
            # Observational: treated + PSID controls
            treated_url = f"{base_url}_treated.txt"
            control_url = "https://users.nber.org/~rdehejia/data/psid_controls.txt"
        elif version == 'cps':
            # Observational: treated + CPS controls
            treated_url = f"{base_url}_treated.txt"
            control_url = "https://users.nber.org/~rdehejia/data/cps_controls.txt"
        else:
            raise ValueError("version must be 'experimental', 'psid', or 'cps'")
        
        # Column names for the dataset
        colnames = ['treat', 'age', 'education', 'black', 'hispanic', 
                    'married', 'nodegree', 're74', 're75', 're78']
        
        # Load treated and control data
        try:
            df_treated = pd.read_csv(treated_url, delim_whitespace=True, 
                                    header=None, names=colnames)
            df_control = pd.read_csv(control_url, delim_whitespace=True, 
                                    header=None, names=colnames)
            
            # Combine
            df = pd.concat([df_treated, df_control], ignore_index=True)
        except:
            print("Loading from URL failed. Using alternative method...")
            # Fallback: load from causaldata package if available
            try:
                from causaldata import lalonde
                df = lalonde.load_pandas().data
                # Standardize column names
                if 'treat' not in df.columns and 'treatment' in df.columns:
                    df = df.rename(columns={'treatment': 'treat', 're78': 're78'})
            except:
                raise Exception("Could not load LaLonde data. Install causaldata: pip install causaldata")
        
        print(f"Loaded LaLonde {version} dataset: {df.shape[0]} units, {df.shape[1]} variables")
        # save to file
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    
    # Define treatment and outcome
    T = df['treat'].values.squeeze()
    Y = df['re78'].values.squeeze()  # Real earnings in 1978
    
    # Select covariates
    if include_categorical:
        # All 8 standard covariates (4 continuous + 4 binary)
        covariate_names = ['age', 'education', 're74', 're75',
                          'black', 'hispanic', 'married', 'nodegree']
    else:
        # Only continuous covariates
        covariate_names = ['age', 'education', 're74', 're75']
    
    X = df[covariate_names].values
    
    # Split by treatment
    control_mask = (T == 0).squeeze()
    treatment_mask = (T == 1).squeeze()
    
    X_control = X[control_mask]
    X_treatment = X[treatment_mask]
    Y_control = Y[control_mask]
    Y_treatment = Y[treatment_mask]
    
    n0 = X_control.shape[0]
    n1 = X_treatment.shape[0]
    d = X.shape[1]
    
    print(f"\n{version.upper()} VERSION:")
    print(f"  Control units: {n0}")
    print(f"  Treatment units: {n1}")
    print(f"  Covariates: {d} ({covariate_names})")
    
    # Calculate naive (confounded) difference
    naive_diff = Y_treatment.mean() - Y_control.mean()
    print(f"\nNaive outcome difference: ${naive_diff:.2f}")
    print(f"  Treatment mean: ${Y_treatment.mean():.2f}")
    print(f"  Control mean: ${Y_control.mean():.2f}")
    
    # True ATT (from RCT experimental benchmark)
    true_ATT = None
    if compute_true_ATT:
        df_exp = pd.read_csv(path.replace('psid', 'experimental'))
         # Define treatment and outcome
        T_exp = df_exp['treat'].values
        Y_exp = df_exp['re78'].values  # Real earnings in 1978

        X_exp = df_exp[covariate_names].values
        # Split by treatment
        control_mask = T_exp == 0
        treatment_mask = T_exp == 1
        
        X_control_exp = X_exp[control_mask]
        X_treatment_exp = X_exp[treatment_mask]
        Y_control_exp = Y_exp[control_mask]
        Y_treatment_exp = Y_exp[treatment_mask]
        true_ATT = Y_treatment_exp.mean() - Y_control_exp.mean()

        print(f"\n✓ TRUE ATT (experimental benchmark): ${true_ATT:.2f}")
        print(f"  (This is the ground truth to compare against)")
    else:
        print(f"\n⚠ This is OBSERVATIONAL data (confounded)")
        print(f"  True ATT from experiment: ~$1794")
        print(f"  Your observational estimate should recover this!")
    
    outcome_info = {
        'name': 're78',
        'description': 'Real earnings in 1978 (dollars)',
        'type': 'continuous',
        'version': version
    }
    
    data = {
        'X_control': X_control,
        'X_treatment': X_treatment,
        'Y_control': Y_control,
        'Y_treatment': Y_treatment,
        'true_ATT': true_ATT,  # based on experimental version
        'Y0_treatment': None,
        'outcome_info': outcome_info,
        'covariate_names': covariate_names,
        'n0': n0,
        'n1': n1,
        'd': d
    }
    print(f"\nOriginal scale statistics:")
    print(f"  Sample: Control={n0}, Treated={n1}, Covariates={d}")
    print(f"\n  Variable scales (mean ± std):")
    for i, var_name in enumerate(covariate_names):
        mean_val = X[:, i].mean()
        std_val = X[:, i].std()
        print(f"    {var_name}: {mean_val:.2f} ± {std_val:.2f}")
    
    print(f"  → Scales differ by {X.std(axis=0).max() / X.std(axis=0).min():.1f}x")
    print(f"  → Note: earnings dominate by ~1000x over indicators!")

    # Standardize if requested
    if standardize:
        print(f"\nStandardizing using {standardize_method} method...")
        data, scaler_params = standardize_data(data, method=standardize_method)
        data['scaler_params'] = scaler_params
        
        X_std = np.vstack([data['X_control'], data['X_treatment']])
        print(f"  After standardization:")
        print(f"    Mean: {X_std.mean(axis=0)}")
        print(f"    Std: {X_std.std(axis=0)}")
    else:
        print("\nNo standardization applied")
        data['scaler_params'] = None
    
    print(f"\nOutcome statistics:")
    print(f"  Control earnings: ${Y_control.mean():.0f}")
    print(f"  Treatment earnings: ${Y_treatment.mean():.0f}")
    print(f"  Naive difference: ${Y_treatment.mean() - Y_control.mean():.0f}")
    
    return data

if __name__ == "__main__":


    # load_lalonde_data(
    #     path="data/lalonde_psid.csv", 
    #     download=True, 
    #     version='psid', 
    #     include_categorical=False,
    #     compute_true_ATT=True,
    #     standardize=True,
    # )


    
    print("="*80)
    print("LALONDE OVERLAP VISUALIZATION")
    print("="*80)
    
    # Load and visualize experimental data (good overlap)
    print("\n1. EXPERIMENTAL DATA (RCT - Good Overlap)")
    print("-"*80)
    data_exp = load_lalonde_data(
        path="data/lalonde_experimental.csv",
        version='experimental',
        include_categorical=False,
        standardize=False
    )
    visualize_lalonde_overlap(data_exp, save_dir="figures/experimental", 
                             version="experimental", show_plots=False)
    
    # Load and visualize PSID data (poor overlap)
    print("\n2. PSID DATA (Observational - Poor Overlap)")
    print("-"*80)
    data_psid = load_lalonde_data(
        path="data/lalonde_psid.csv",
        version='psid',
        include_categorical=False,
        standardize=False
    )
    visualize_lalonde_overlap(data_psid, save_dir="figures/psid",
                             version="psid", show_plots=False)
    
    print("\n" + "="*80)
    print("✓ All visualizations complete!")
    print("="*80)

    