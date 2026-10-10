"""
MEPS Medicare ATT Analysis - Data Processing Pipeline
======================================================

Processes MEPS Full Year Consolidated (FYC) data for 2019-2022 to estimate
the Average Treatment Effect on the Treated (ATT) for Medicare coverage on
total healthcare expenditure.

Experiment Design:
- Treatment: Binary indicator for Medicare coverage (MCREV=1)
- Outcome: Total healthcare expenditure (TOTEXP)
- Confounders: Age, income (% poverty line), health status (PCS-42, MCS-42, PHQ-2),
  prior healthcare utilization, self-reported health, chronic conditions count
- Sample: Individuals aged 58-72 with clean Medicare eligibility
  - Treated: Age >= 65 with Medicare (MCREV=1) - age-based eligibility
  - Control: Age < 65 without Medicare (MCREV=2)
  - Excluded: Age < 65 with Medicare (disability pathway)
  - Excluded: Age >= 65 without Medicare (non-compliers)

This design creates extreme lack of overlap by construction (~95% jump in coverage
probability at age 65).

Author: Generated for Yidan's causal inference research
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')
import os
import json

# ==============================================================================
# CONFIGURATION
# ==============================================================================

FILE_MAP = {
    2019: 'h216.dta',
    2020: 'h224.dta',
    2021: 'h233.dta',
    2022: 'h243.dta'
}

AGE_MIN = 58
AGE_MAX = 72
MEDICARE_ELIGIBILITY_AGE = 65

CORE_VARS = {
    'DUPERSID': 'person_id',
    'SEX': 'sex',
    'RACETHX': 'race_ethnicity',
    'REGION': 'region',
    'MARRY': 'marital_status',
    'TOTEXP': 'total_exp',
    'MCREV': 'medicare_ever',
    'POVLEV': 'pov_level',
    'POVCAT': 'pov_category',
    'FAMINC': 'family_income',
    'PCS42': 'pcs42',
    'MCS42': 'mcs42',
    'PHQ242': 'phq2',
    'RTHLTH42': 'health_status',
    'OBTOTV': 'office_visits',
    'OPTOTV': 'outpatient_visits',
    'ERTOT': 'er_visits',
    'IPDIS': 'inpatient_stays',
    'RXTOT': 'rx_total',
    'HIBPDX': 'has_hypertension',
    'CHDDX': 'has_chd',
    'DIABDX': 'has_diabetes',
    'ASTHDX': 'has_asthma',
    'PERWT': 'weight',
}

AGE_VARS = {
    2019: 'AGE19X',
    2020: 'AGE20X',
    2021: 'AGE21X',
    2022: 'AGE22X'
}

ALT_VAR_NAMES = {
    'POVLEV': ['POVLEV19', 'POVLEV20', 'POVLEV21', 'POVLEV22'],
    'PCS42': ['PCS42', 'SFPCS42'],
    'MCS42': ['MCS42', 'SFMCS42'],
    'PHQ242': ['PHQ242', 'K6SUM42'],
    'PERWT': ['PERWT19F', 'PERWT20F', 'PERWT21F', 'PERWT22F', 'PERWTF'],
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_meps_stata_files(data_dir='meps_data', years=[2019, 2020, 2021, 2022]):
    """Load MEPS Stata files for specified years."""
    data_dir = Path(data_dir)
    dfs = {}
    
    print("="*70)
    print("LOADING MEPS DATA")
    print("="*70)
    
    for year in years:
        filename = FILE_MAP.get(year)
        if filename is None:
            print(f"⚠ Warning: No file mapping for year {year}")
            continue
            
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"⚠ Warning: File not found: {filepath}")
            continue
            
        print(f"\nLoading {year} data from {filename}...")
        try:
            df = pd.read_stata(filepath, convert_categoricals=False)
            print(f"  ✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
            dfs[year] = df
        except Exception as e:
            print(f"  ✗ Error loading {filepath}: {e}")
    
    return dfs


# ==============================================================================
# VARIABLE EXTRACTION
# ==============================================================================

def find_variable(df, primary_name, alternatives=None, year=None):
    """Find a variable in the dataframe, trying alternatives if primary not found."""
    cols_upper = {c.upper(): c for c in df.columns}
    
    if primary_name.upper() in cols_upper:
        return cols_upper[primary_name.upper()], True
    
    if alternatives:
        for alt in alternatives:
            if alt.upper() in cols_upper:
                return cols_upper[alt.upper()], True
    
    if year:
        year_suffixes = [str(year)[-2:], str(year), f'{year}X']
        for suffix in year_suffixes:
            test_name = f"{primary_name}{suffix}".upper()
            if test_name in cols_upper:
                return cols_upper[test_name], True
    
    return None, False


def extract_variables(df, year):
    """Extract and standardize variables from MEPS DataFrame."""
    print(f"\n  Extracting variables for {year}...")
    
    result = pd.DataFrame()
    result['year'] = year
    missing_vars = []
    
    # Extract age (year-specific)
    age_var = AGE_VARS.get(year)
    if age_var:
        col_name, found = find_variable(df, age_var, year=year)
        if found:
            result['age'] = df[col_name].copy()
            print(f"    ✓ Age: {col_name}")
        else:
            col_name, found = find_variable(df, 'AGE', year=year)
            if found:
                result['age'] = df[col_name].copy()
                print(f"    ✓ Age: {col_name}")
            else:
                missing_vars.append('AGE')
    
    # Extract core variables
    for meps_var, std_name in CORE_VARS.items():
        alternatives = ALT_VAR_NAMES.get(meps_var, [])
        col_name, found = find_variable(df, meps_var, alternatives, year)
        
        if found:
            result[std_name] = df[col_name].copy()
            print(f"    ✓ {std_name}: {col_name}")
        else:
            missing_vars.append(meps_var)
            result[std_name] = np.nan
    
    if missing_vars:
        print(f"    ⚠ Missing variables: {missing_vars}")
    
    return result


# ==============================================================================
# SAMPLE CONSTRUCTION
# ==============================================================================

def construct_medicare_att_sample(df, verbose=True):
    """Construct the analysis sample following the experiment design."""
    if verbose:
        print("\n" + "="*70)
        print("CONSTRUCTING ANALYSIS SAMPLE")
        print("="*70)
        print(f"\nInitial sample: {len(df):,} observations")
    
    # Step 1: Restrict to age range 58-72
    df_age = df[(df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX)].copy()
    if verbose:
        print(f"\nAfter age restriction ({AGE_MIN}-{AGE_MAX}): {len(df_age):,} observations")
        print(f"  Age distribution: mean={df_age['age'].mean():.1f}, "
              f"min={df_age['age'].min():.0f}, max={df_age['age'].max():.0f}")
    
    # Step 2: Clean Medicare variable
    df_clean = df_age[df_age['medicare_ever'].isin([1, 2])].copy()
    if verbose:
        n_missing_medicare = len(df_age) - len(df_clean)
        print(f"\nAfter removing missing Medicare status: {len(df_clean):,} observations")
        print(f"  Removed {n_missing_medicare:,} with missing/inapplicable Medicare status")
    
    df_clean['has_medicare'] = (df_clean['medicare_ever'] == 1).astype(int)
    
    # Step 3: Apply sample restrictions
    under65_no_medicare = (df_clean['age'] < MEDICARE_ELIGIBILITY_AGE) & (df_clean['has_medicare'] == 0)
    under65_with_medicare = (df_clean['age'] < MEDICARE_ELIGIBILITY_AGE) & (df_clean['has_medicare'] == 1)
    over65_with_medicare = (df_clean['age'] >= MEDICARE_ELIGIBILITY_AGE) & (df_clean['has_medicare'] == 1)
    over65_no_medicare = (df_clean['age'] >= MEDICARE_ELIGIBILITY_AGE) & (df_clean['has_medicare'] == 0)
    
    if verbose:
        print(f"\nSample breakdown by age and Medicare status:")
        print(f"  Age < 65, No Medicare (CONTROL):    {under65_no_medicare.sum():,}")
        print(f"  Age < 65, With Medicare (EXCLUDED): {under65_with_medicare.sum():,}  <- disability pathway")
        print(f"  Age >= 65, With Medicare (TREATED): {over65_with_medicare.sum():,}")
        print(f"  Age >= 65, No Medicare (EXCLUDED):  {over65_no_medicare.sum():,}  <- non-compliers")
    
    df_analysis = df_clean[under65_no_medicare | over65_with_medicare].copy()
    df_analysis['treatment'] = (df_analysis['age'] >= MEDICARE_ELIGIBILITY_AGE).astype(int)
    
    if verbose:
        print(f"\nFinal analysis sample: {len(df_analysis):,} observations")
        print(f"  Treated (Age >= 65 with Medicare): {df_analysis['treatment'].sum():,} "
              f"({100*df_analysis['treatment'].mean():.1f}%)")
        print(f"  Control (Age < 65 without Medicare): {(1-df_analysis['treatment']).sum():,} "
              f"({100*(1-df_analysis['treatment'].mean()):.1f}%)")
    
    return df_analysis


# =============================================================================
# OVERLAP VISUALIZATION
# =============================================================================

def plot_overlap(data, save_path=None, show=True):
    """
    Plot covariate distributions to visualize overlap between treated and control.
    
    Parameters:
    -----------
    data : dict
        Data dictionary from load_meps_data()
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    """
    import matplotlib.pyplot as plt
    
    X_control = data['X_control']
    X_treatment = data['X_treatment']
    covariate_names = data['covariate_names']
    n_covariates = len(covariate_names)
    
    # Calculate number of rows needed (3 columns)
    n_cols = 3
    n_rows = (n_covariates + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, var_name in enumerate(covariate_names):
        ax = axes[i]
        
        # Get data for this covariate
        x_ctrl = X_control[:, i]
        x_treat = X_treatment[:, i]
        
        # Calculate SMD
        mean_diff = x_treat.mean() - x_ctrl.mean()
        pooled_std = np.sqrt((x_treat.var() + x_ctrl.var()) / 2)
        smd = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Plot histograms
        bins = 30
        ax.hist(x_ctrl, bins=bins, alpha=0.5, label=f'Control (n={len(x_ctrl)})', 
                density=True, color='blue')
        ax.hist(x_treat, bins=bins, alpha=0.5, label=f'Treated (n={len(x_treat)})', 
                density=True, color='red')
        
        # Add vertical lines for means
        ax.axvline(x_ctrl.mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(x_treat.mean(), color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel(var_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{var_name}\nSMD = {smd:.2f}')
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(n_covariates, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Covariate Overlap: Control vs Treated', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved overlap plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_propensity_overlap(data, save_path=None, show=True):
    """
    Plot propensity score distributions (estimated via logistic regression).
    
    Parameters:
    -----------
    data : dict
        Data dictionary from load_meps_data()
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    """
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    
    # Combine data
    X = np.vstack([data['X_control'], data['X_treatment']])
    T = np.concatenate([np.zeros(data['n0']), np.ones(data['n1'])])
    
    # Fit propensity score model
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    
    ps_control = propensity_scores[T == 0]
    ps_treated = propensity_scores[T == 1]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(ps_control, bins=50, alpha=0.5, label=f'Control (n={len(ps_control)})', 
             density=True, color='blue')
    ax1.hist(ps_treated, bins=50, alpha=0.5, label=f'Treated (n={len(ps_treated)})', 
             density=True, color='red')
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distribution')
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([ps_control, ps_treated], labels=['Control', 'Treated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.5)
    ax2.set_ylabel('Propensity Score')
    ax2.set_title('Propensity Score by Treatment Status')
    
    # Add overlap statistics
    overlap_region = (max(ps_control.min(), ps_treated.min()), 
                      min(ps_control.max(), ps_treated.max()))
    ctrl_in_overlap = ((ps_control >= overlap_region[0]) & (ps_control <= overlap_region[1])).mean()
    treat_in_overlap = ((ps_treated >= overlap_region[0]) & (ps_treated <= overlap_region[1])).mean()
    
    textstr = f'Overlap region: [{overlap_region[0]:.3f}, {overlap_region[1]:.3f}]\n'
    textstr += f'Control in overlap: {ctrl_in_overlap:.1%}\n'
    textstr += f'Treated in overlap: {treat_in_overlap:.1%}'
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Propensity Score Analysis (Logistic Regression)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved propensity plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, propensity_scores


def plot_smd_summary(data, save_path=None, show=True):
    """
    Plot standardized mean differences as a bar chart.
    
    Parameters:
    -----------
    data : dict
        Data dictionary from load_meps_data()
    save_path : str
        Path to save figure (optional)
    show : bool
        Whether to display the plot
    """
    import matplotlib.pyplot as plt
    
    X_control = data['X_control']
    X_treatment = data['X_treatment']
    covariate_names = data['covariate_names']
    
    # Calculate SMDs
    smds = []
    for i in range(len(covariate_names)):
        x_ctrl = X_control[:, i]
        x_treat = X_treatment[:, i]
        mean_diff = x_treat.mean() - x_ctrl.mean()
        pooled_std = np.sqrt((x_treat.var() + x_ctrl.var()) / 2)
        smd = mean_diff / pooled_std if pooled_std > 0 else 0
        smds.append(smd)
    
    # Sort by absolute SMD
    sorted_idx = np.argsort(np.abs(smds))[::-1]
    sorted_names = [covariate_names[i] for i in sorted_idx]
    sorted_smds = [smds[i] for i in sorted_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(covariate_names) * 0.5)))
    
    colors = ['red' if abs(s) > 0.25 else 'orange' if abs(s) > 0.1 else 'green' for s in sorted_smds]
    bars = ax.barh(range(len(sorted_names)), sorted_smds, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Standardized Mean Difference (SMD)')
    ax.set_title('Covariate Balance: SMD Summary')
    
    # Add reference lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=-0.1, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0.25, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=-0.25, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='|SMD| ≤ 0.1 (balanced)'),
        Patch(facecolor='orange', alpha=0.7, label='0.1 < |SMD| ≤ 0.25 (mild imbalance)'),
        Patch(facecolor='red', alpha=0.7, label='|SMD| > 0.25 (serious imbalance)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved SMD plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, dict(zip(covariate_names, smds))

# ==============================================================================
# COVARIATE PROCESSING
# ==============================================================================

def process_covariates(df, verbose=True, drop_threshold=0.05):
    """
    Process and clean covariates for analysis.
    
    Missing data handling:
    - Columns with missingness > drop_threshold (5%): DROP the column
    - This ensures a clean sample without imputation complexity
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analysis sample
    verbose : bool
        Print processing details
    drop_threshold : float
        Drop columns with missingness above this threshold (default: 0.05 = 5%)
    """
    if verbose:
        print("\n" + "="*70)
        print("PROCESSING COVARIATES")
        print("="*70)
        print(f"\nRule: Drop COLUMN if missingness > {drop_threshold:.0%}")
    
    result = df.copy()
    
    # Define all potential confounders
    confounders = {
        'age': 'Age (years)',
        'pov_level': 'Income (% poverty line)',
        'pcs42': 'Physical health (PCS-42)',
        'mcs42': 'Mental health (MCS-42)',
        'phq2': 'Depression (PHQ-2)',
        'office_visits': 'Office visits',
        'outpatient_visits': 'Outpatient visits',
        'er_visits': 'ER visits',
        'inpatient_stays': 'Inpatient stays',
        'rx_total': 'Prescription fills',
        'health_status': 'Self-reported health (1-5)',
    }
    
    cols_to_drop = []
    cols_to_keep = []
    
    if verbose:
        print(f"\n{'Variable':<35} {'Valid':>8} {'Missing':>8} {'%Miss':>8} {'Action':<12}")
        print("-" * 75)
    
    # Assess each confounder
    for var, label in confounders.items():
        if var not in result.columns:
            if verbose:
                print(f"{label:<35} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'NOT IN DATA':<12}")
            continue
        
        # Convert negative values to NaN (MEPS coding for missing)
        result.loc[result[var] < 0, var] = np.nan
        
        n_total = len(result)
        n_missing = result[var].isna().sum()
        n_valid = n_total - n_missing
        pct_missing = n_missing / n_total
        
        if pct_missing > drop_threshold:
            action = "DROP"
            cols_to_drop.append(var)
        else:
            action = "KEEP"
            cols_to_keep.append(var)
        
        if verbose:
            print(f"{label:<35} {n_valid:>8,} {n_missing:>8,} {pct_missing:>7.1%} {action:<12}")
    
    # Drop columns with high missingness
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
        if verbose:
            print(f"\n  Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    
    if verbose:
        print(f"  Keeping {len(cols_to_keep)} columns: {cols_to_keep}")
    
    # Create chronic conditions count
    chronic_vars = ['has_hypertension', 'has_chd', 'has_diabetes', 'has_asthma']
    available_chronic = [v for v in chronic_vars if v in result.columns]
    
    if available_chronic:
        for var in available_chronic:
            result.loc[result[var] < 0, var] = 0
            result[var] = (result[var] == 1).astype(int)
        
        result['chronic_conditions'] = result[available_chronic].sum(axis=1)
        if verbose:
            print(f"\n  Created 'chronic_conditions' from {available_chronic}")
            print(f"    Range: [{result['chronic_conditions'].min():.0f}, "
                  f"{result['chronic_conditions'].max():.0f}], "
                  f"Mean: {result['chronic_conditions'].mean():.2f}")
    
    # Handle outcome - drop rows only for missing outcome
    if 'total_exp' in result.columns:
        result.loc[result['total_exp'] < 0, 'total_exp'] = np.nan
        n_before = len(result)
        result = result.dropna(subset=['total_exp'])
        n_after = len(result)
        if n_before > n_after and verbose:
            print(f"\n  Dropped {n_before - n_after:,} rows with missing outcome")
    
    if verbose:
        print(f"\n  Final sample size: {len(result):,}")
        print(f"  Final covariates: {cols_to_keep + ['chronic_conditions']}")
    
    return result


def calculate_overlap_diagnostics(df, verbose=True):
    """Calculate diagnostics for covariate overlap between treated and control."""
    if verbose:
        print("\n" + "="*70)
        print("OVERLAP DIAGNOSTICS")
        print("="*70)
    
    treated = df[df['treatment'] == 1]
    control = df[df['treatment'] == 0]
    
    confounders = ['age', 'pov_level', 'pcs42', 'mcs42', 'phq2',
                   'office_visits', 'outpatient_visits', 'er_visits',
                   'inpatient_stays', 'rx_total', 'health_status', 'chronic_conditions']
    
    diagnostics = {}
    
    if verbose:
        print(f"\nCovariate Balance (Standardized Mean Differences):")
        print("-" * 60)
        print(f"{'Variable':<25} {'SMD':>10} {'Control Mean':>15} {'Treated Mean':>15}")
        print("-" * 60)
    
    for var in confounders:
        if var not in df.columns:
            continue
        
        mean_t = treated[var].mean()
        mean_c = control[var].mean()
        var_t = treated[var].var()
        var_c = control[var].var()
        n_t = treated[var].notna().sum()
        n_c = control[var].notna().sum()
        
        if n_t > 0 and n_c > 0:
            pooled_std = np.sqrt((var_t + var_c) / 2)
            smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0
        else:
            smd = np.nan
        
        diagnostics[var] = {
            'smd': smd,
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'n_treated': n_t,
            'n_control': n_c
        }
        
        if verbose and not np.isnan(smd):
            print(f"{var:<25} {smd:>10.3f} {mean_c:>15.2f} {mean_t:>15.2f}")
    
    if verbose:
        print("-" * 60)
    
    return diagnostics


def process_outcome(df, verbose=True):
    """Process the outcome variable (total healthcare expenditure)."""
    if verbose:
        print("\n" + "="*70)
        print("PROCESSING OUTCOME: Total Healthcare Expenditure")
        print("="*70)
    
    result = df.copy()
    result.loc[result['total_exp'] < 0, 'total_exp'] = np.nan
    
    treated = result[result['treatment'] == 1]
    control = result[result['treatment'] == 0]
    
    if verbose:
        print(f"\nOutcome Distribution by Treatment Status:")
        print("-" * 50)
        print(f"{'Statistic':<20} {'Control':>15} {'Treated':>15}")
        print("-" * 50)
        print(f"{'N':<20} {len(control):>15,} {len(treated):>15,}")
        print(f"{'Mean':<20} ${control['total_exp'].mean():>14,.0f} ${treated['total_exp'].mean():>14,.0f}")
        print(f"{'Median':<20} ${control['total_exp'].median():>14,.0f} ${treated['total_exp'].median():>14,.0f}")
        print(f"{'Std Dev':<20} ${control['total_exp'].std():>14,.0f} ${treated['total_exp'].std():>14,.0f}")
        print("-" * 50)
        
        raw_diff = treated['total_exp'].mean() - control['total_exp'].mean()
        print(f"\nRaw Mean Difference (Treated - Control): ${raw_diff:,.0f}")
    
    return result


# ==============================================================================
# MAIN PROCESSING PIPELINE
# ==============================================================================

def process_meps_for_att_analysis(data_dir='meps_data', 
                                   year=2022,
                                   save_output=True,
                                   output_dir='processed_data'):
    """Main pipeline to process MEPS data for Medicare ATT analysis."""
    years = [year] if isinstance(year, int) else year
    
    print("\n" + "="*70)
    print("MEPS MEDICARE ATT ANALYSIS - DATA PROCESSING PIPELINE")
    print("="*70)
    print(f"\nUsing single year: {year} (to avoid repeated individuals)")
    
    raw_dfs = load_meps_stata_files(data_dir, years)
    if not raw_dfs:
        raise ValueError("No data files loaded!")
    
    print("\n" + "="*70)
    print("EXTRACTING VARIABLES")
    print("="*70)
    
    processed_dfs = []
    for yr, df in raw_dfs.items():
        processed = extract_variables(df, yr)
        processed_dfs.append(processed)
    
    df_combined = pd.concat(processed_dfs, ignore_index=True)
    print(f"\nCombined data: {len(df_combined):,} observations")
    
    df_sample = construct_medicare_att_sample(df_combined, verbose=True)
    df_covariates = process_covariates(df_sample, verbose=True)
    diagnostics = calculate_overlap_diagnostics(df_covariates, verbose=True)
    df_final = process_outcome(df_covariates, verbose=True)
    
    summary = {
        'n_total': len(df_final),
        'n_treated': df_final['treatment'].sum(),
        'n_control': len(df_final) - df_final['treatment'].sum(),
        'pct_treated': 100 * df_final['treatment'].mean(),
        'year': year,
        'diagnostics': diagnostics
    }
    
    if save_output:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / 'meps_medicare_att_sample.csv'
        df_final.to_csv(output_file, index=False)
        print(f"\n✓ Saved analysis sample to: {output_file}")
        
        summary_file = output_path / 'processing_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("MEPS Medicare ATT Analysis - Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total observations: {summary['n_total']:,}\n")
            f.write(f"Treated: {summary['n_treated']:,} ({summary['pct_treated']:.1f}%)\n")
            f.write(f"Control: {summary['n_control']:,} ({100-summary['pct_treated']:.1f}%)\n")
            f.write(f"Year: {summary['year']}\n\n")
            f.write("Covariate Balance (SMD):\n")
            for var, diag in diagnostics.items():
                f.write(f"  {var}: {diag['smd']:.3f}\n")
        print(f"✓ Saved summary to: {summary_file}")
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    
    return df_final, summary


# ==============================================================================
# DATA LOADER (mirrors load_rhc_data interface)
# ==============================================================================

def standardize_data(data, method='zscore'):
    """Standardize covariates."""
    X_all = np.vstack([data['X_control'], data['X_treatment']])
    
    if method == 'zscore':
        mean = X_all.mean(axis=0)
        std = X_all.std(axis=0)
        std[std == 0] = 1
        data['X_control'] = (data['X_control'] - mean) / std
        data['X_treatment'] = (data['X_treatment'] - mean) / std
        scaler_params = {'method': 'zscore', 'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = X_all.min(axis=0)
        max_val = X_all.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        data['X_control'] = (data['X_control'] - min_val) / range_val
        data['X_treatment'] = (data['X_treatment'] - min_val) / range_val
        scaler_params = {'method': 'minmax', 'min': min_val, 'max': max_val}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return data, scaler_params


def load_meps_data(
    raw_data_dir: str = "meps_data",
    processed_path: str = "meps_data/processed/meps_medicare_att.csv",
    year: int = 2022,
    process: bool = False,
    standardize: bool = True,
    standardize_method: str = 'zscore',
    semisynthetic: bool = False,
    semisynthetic_path: str = None,
):
    """
    Load MEPS Medicare data and prepare for causal inference.
    
    Parameters:
    -----------
    raw_data_dir : str
        Directory containing raw MEPS .dta files
    processed_path : str
        Path for processed CSV file
    year : int
        Year to use (default: 2022)
    process : bool
        If True, process raw data even if processed exists
    standardize : bool
        Whether to standardize covariates
    standardize_method : str
        'zscore' or 'minmax'
    semisynthetic : bool
        If True, load semi-synthetic data with known ground truth
    semisynthetic_path : str
        Path to semi-synthetic data file
        
    Returns:
    --------
    data : dict
        Dictionary containing X_control, X_treatment, Y_control, Y_treatment, etc.
    """
    metadata = None
    
    treat_var = 'treatment'
    outcome_var = 'total_exp'
    outcome_info = {
        'name': 'total_exp',
        'description': 'Total healthcare expenditure ($)',
        'type': 'positive'
    }
    
    # Updated confounder list with additional variables
    confounder_vars = [
        'age', 'pov_level', 'pcs42', 'mcs42', 'phq2',
        'office_visits', 'outpatient_visits', 'er_visits',
        'inpatient_stays', 'rx_total',
        'health_status', 'chronic_conditions'
    ]
    
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PROCESS RAW DATA IF NEEDED
    # =========================================================================
    if process or not os.path.exists(processed_path):
        print("="*70)
        print(f"PROCESSING MEPS DATA (Year: {year})")
        print("="*70)
        
        raw_dfs = load_meps_stata_files(raw_data_dir, years=[year])
        if not raw_dfs:
            raise FileNotFoundError(f"No MEPS data found in {raw_data_dir}")
        
        df = extract_variables(raw_dfs[year], year)
        df = construct_medicare_att_sample(df, verbose=True)
        df = process_covariates(df, verbose=True)
        
        df.to_csv(processed_path, index=False)
        print(f"\n✓ Saved processed data to: {processed_path}")
        prep_df = df
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    else:
        if semisynthetic:
            if semisynthetic_path is None:
                semisynthetic_path = processed_path.replace('.csv', '_semisynthetic.csv')
            
            if not os.path.exists(semisynthetic_path):
                raise FileNotFoundError(
                    f"Semi-synthetic data not found at {semisynthetic_path}. "
                    f"Generate it first using create_semisynthetic_meps()."
                )
            
            print(f"\nLoading SEMI-SYNTHETIC data from: {semisynthetic_path}")
            prep_df = pd.read_csv(semisynthetic_path)
            
            try:
                metadata_path = semisynthetic_path.replace('.csv', '_metadata.json')
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                true_effects = metadata.get('true_effects', {})
                generation_params = metadata.get('generation_params', {})
                
                print(f"\n{'='*60}")
                print("SEMI-SYNTHETIC DATA INFO")
                print('='*60)
                print(f"  True ATT: {true_effects.get('ATT', 'N/A')}")
                print(f"  True ATE: {true_effects.get('ATE', 'N/A')}")
                if generation_params:
                    print(f"  Outcome type: {generation_params.get('outcome_type', 'N/A')}")
                print('='*60)
                
            except Exception as e:
                print(f"  Warning: Could not load metadata: {e}")
                metadata = None
            
            treat_var = 'treatment_semi' if 'treatment_semi' in prep_df.columns else 'treatment'
            outcome_var = 'total_exp_semi' if 'total_exp_semi' in prep_df.columns else 'total_exp'
            
        else:
            print(f"\nLoading ORIGINAL observational data from: {processed_path}")
            prep_df = pd.read_csv(processed_path)
    
    # =========================================================================
    # EXTRACT DATA
    # =========================================================================
    
    available_confounders = [v for v in confounder_vars if v in prep_df.columns]
    
    # Only drop rows with missing outcome or treatment
    essential_cols = [treat_var, outcome_var]
    prep_df_clean = prep_df.dropna(subset=[c for c in essential_cols if c in prep_df.columns])
    
    n_dropped = len(prep_df) - len(prep_df_clean)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} rows with missing outcome/treatment")
    
    # Impute remaining missing covariate values with the median of each variable
    if available_confounders:
        print("  Imputing missing covariate values with variable-wise medians")
        for var in available_confounders:
            if var in prep_df_clean.columns and pd.api.types.is_numeric_dtype(prep_df_clean[var]):
                median_val = prep_df_clean[var].median()
                if pd.isna(median_val):
                    # If the variable is entirely missing, leave as-is
                    continue
                prep_df_clean[var] = prep_df_clean[var].fillna(median_val)
    
    X = prep_df_clean[available_confounders].values
    T = prep_df_clean[treat_var].values
    Y = prep_df_clean[outcome_var].values.astype(float)
    
    control_mask = (T == 0)
    treatment_mask = (T == 1)
    
    X_control = X[control_mask]
    X_treatment = X[treatment_mask]
    Y_control = Y[control_mask]
    Y_treatment = Y[treatment_mask]
    
    n0, n1, d = X_control.shape[0], X_treatment.shape[0], X.shape[1]
    
    print(f"\n  Extracted data:")
    print(f"    X shape: {X.shape}")
    print(f"    Covariates: {available_confounders}")
    print(f"    T: {n1:,} treated, {n0:,} control")
    print(f"    Y: range=[${Y.min():,.0f}, ${Y.max():,.0f}], mean=${Y.mean():,.0f}")
    
    print(f"\nObserved outcomes:")
    print(f"  Control: ${Y_control.mean():,.0f}")
    print(f"  Treatment: ${Y_treatment.mean():,.0f}")
    print(f"  Naive difference: ${Y_treatment.mean() - Y_control.mean():,.0f}")
    
    if metadata is not None and 'true_effects' in metadata:
        true_att = metadata['true_effects'].get('ATT')
        if true_att is not None:
            selection_bias = (Y_treatment.mean() - Y_control.mean()) - true_att
            print(f"  True ATT: ${true_att:,.0f}")
            print(f"  → Selection bias: ${selection_bias:,.0f}")
    
    # =========================================================================
    # CREATE DATA DICTIONARY
    # =========================================================================
    
    data = {
        'X_control': X_control,
        'X_treatment': X_treatment,
        'Y_control': Y_control,
        'Y_treatment': Y_treatment,
        'covariate_names': available_confounders,
        'outcome_info': outcome_info,
        'true_ATT': None,
        'true_ATE': None,
        'Y0_treatment': None,
        'Y1_treatment': None,
        'n0': n0,
        'n1': n1,
        'd': d,
    }
    
    if metadata is not None:
        true_effects = metadata.get('true_effects', {})
        data['true_ATT'] = true_effects.get('ATT')
        data['true_ATE'] = true_effects.get('ATE')
        data['metadata'] = metadata
        
        if 'Y0' in prep_df_clean.columns and 'Y1' in prep_df_clean.columns:
            data['Y0_treatment'] = prep_df_clean.loc[treatment_mask, 'Y0'].values
            data['Y1_treatment'] = prep_df_clean.loc[treatment_mask, 'Y1'].values
            
            computed_att = (data['Y1_treatment'] - data['Y0_treatment']).mean()
            print(f"\n  ✓ Verification from counterfactuals: ATT = ${computed_att:,.0f}")
    
    if standardize:
        print(f"\nStandardizing using {standardize_method} method...")
        data, scaler_params = standardize_data(data, method=standardize_method)
        data['scaler_params'] = scaler_params
        
        X_std = np.vstack([data['X_control'], data['X_treatment']])
        print(f"  After: mean={X_std.mean(axis=0).mean():.3f}, std={X_std.std(axis=0).mean():.3f}")
    else:
        data['scaler_params'] = None
    
    return data


# =============================================================================
# SEMI-SYNTHETIC DATA GENERATION
# =============================================================================
def create_semisynthetic_meps(
    processed_path: str = "meps_data/processed/meps_medicare_att.csv",
    output_path: str = None,
    outcome_type: str = 'linear',
    base_effect: float = 2000.0,
    heterogeneous: bool = True,
    age_confounding: str = 'severe',  # 'mild', 'moderate', 'severe', 'extreme'
    seed: int = 42,
):
    """
    Create semi-synthetic MEPS data with known ground truth ATT.
    
    Parameters:
    -----------
    processed_path : str
        Path to processed MEPS data
    output_path : str
        Output path (default: adds _semisynthetic suffix)
    outcome_type : str
        'linear' or 'nonlinear'
    base_effect : float
        Base treatment effect in dollars
    heterogeneous : bool
        If True, treatment effect varies with age
    age_confounding : str
        Severity of age-based confounding:
        - 'mild': age has small weight in outcome
        - 'moderate': age has moderate weight
        - 'severe': age dominates the outcome (default)
        - 'extreme': age almost entirely determines outcome
    seed : int
        Random seed
        
    The key insight: In MEPS Medicare data, treatment is determined by age>=65.
    By making Y(0) heavily depend on age, we create severe confounding where
    the treated (old) and control (young) have very different baseline outcomes.
    This is the "lack of overlap" problem - no young people look like old people.
    """
    np.random.seed(seed)
    
    if output_path is None:
        output_path = processed_path.replace('.csv', f'_semisynthetic_{age_confounding}.csv')
    
    print("="*70)
    print("CREATING SEMI-SYNTHETIC MEPS DATA")
    print("="*70)
    
    # Load processed data
    df = pd.read_csv(processed_path)
    print(f"Loaded {len(df):,} observations")
    
    confounder_vars = ['age', 'pov_level', 'pcs42', 'mcs42', 'phq2',
                       'office_visits', 'outpatient_visits', 'er_visits',
                       'inpatient_stays', 'rx_total', 'health_status', 'chronic_conditions']
    available = [v for v in confounder_vars if v in df.columns]
    
    df_clean = df.dropna(subset=available + ['treatment', 'total_exp']).copy()
    
    X = df_clean[available].values
    T = df_clean['treatment'].values
    n = len(df_clean)
    
    print(f"After dropping missing: {n:,} observations")
    print(f"Covariates: {available}")
    print(f"\nConfounding severity: {age_confounding.upper()}")
    
    # Standardize covariates
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_std_normalized = (X - X_mean) / X_std
    
    # Get age index
    age_idx = available.index('age')
    age_std = X_std_normalized[:, age_idx]
    
    # Define confounding weights based on severity
    # Higher age_weight = more confounding = harder to estimate ATT
    confounding_params = {
        'mild': {'age_weight': 500, 'other_weight': 200, 'noise': 3000},
        'moderate': {'age_weight': 1500, 'other_weight': 150, 'noise': 2500},
        'severe': {'age_weight': 3000, 'other_weight': 100, 'noise': 2000},
        'extreme': {'age_weight': 5000, 'other_weight': 50, 'noise': 1500},
    }
    
    params = confounding_params[age_confounding]
    age_weight = params['age_weight']
    other_weight = params['other_weight']
    noise_std = params['noise']
    
    print(f"  Age weight in Y(0): {age_weight}")
    print(f"  Other covariate weight: {other_weight}")
    print(f"  Noise std: {noise_std}")
    
    # =========================================================================
    # GENERATE POTENTIAL OUTCOMES
    # =========================================================================
    
    # Baseline intercept
    baseline = 5000
    
    if outcome_type == 'linear':
        # Y(0) = baseline + age_weight * age + other_weight * (other covariates) + noise
        # Age dominates, creating severe confounding
        
        # Age effect (main source of confounding)
        age_effect = age_weight * age_std
        
        # Other covariates have smaller effects
        # Signs chosen to be realistic (worse health = higher spending)
        other_effects = np.zeros(n)
        for i, var in enumerate(available):
            if var == 'age':
                continue  # Already handled
            elif var in ['pcs42', 'mcs42']:
                # Better health (higher score) = lower spending
                other_effects += -other_weight * 0.5 * X_std_normalized[:, i]
            elif var in ['office_visits', 'outpatient_visits', 'er_visits', 
                        'inpatient_stays', 'rx_total', 'chronic_conditions']:
                # More utilization = higher spending
                other_effects += other_weight * X_std_normalized[:, i]
            elif var == 'pov_level':
                # Higher income = slightly lower spending (better access to preventive care)
                other_effects += -other_weight * 0.3 * X_std_normalized[:, i]
            elif var == 'phq2':
                # Depression = higher spending
                other_effects += other_weight * 0.5 * X_std_normalized[:, i]
        
        Y0 = baseline + age_effect + other_effects + np.random.normal(0, noise_std, n)
        
    else:  # nonlinear
        # Nonlinear: age has quadratic effect + interactions
        age_effect = age_weight * (age_std + 0.3 * age_std**2)
        
        # Interaction: age amplifies effect of chronic conditions
        if 'chronic_conditions' in available:
            chronic_idx = available.index('chronic_conditions')
            interaction = 500 * age_std * X_std_normalized[:, chronic_idx]
        else:
            interaction = 0
        
        other_effects = other_weight * X_std_normalized.sum(axis=1)
        
        Y0 = baseline + age_effect + other_effects + interaction + np.random.normal(0, noise_std, n)
    
    Y0 = np.clip(Y0, 0, None)  # Expenditure is non-negative
    
    # =========================================================================
    # TREATMENT EFFECT
    # =========================================================================
    
    if heterogeneous:
        # Treatment effect also depends on age (older patients benefit more)
        # This creates ATT ≠ ATE
        tau = base_effect + 800 * age_std
        
        # Can also add interaction with health status
        if 'pcs42' in available:
            pcs_idx = available.index('pcs42')
            # Sicker patients (lower PCS) benefit more
            tau += -300 * X_std_normalized[:, pcs_idx]
    else:
        tau = np.full(n, base_effect)
    
    Y1 = Y0 + tau
    Y1 = np.clip(Y1, 0, None)
    
    # Observed outcome
    Y_obs = np.where(T == 1, Y1, Y0)
    
    # =========================================================================
    # COMPUTE TRUE EFFECTS
    # =========================================================================
    
    true_ATT = tau[T == 1].mean()
    true_ATE = tau.mean()
    true_ATU = tau[T == 0].mean()  # ATU for completeness
    
    # Compute selection bias
    # Selection bias = E[Y0 | T=1] - E[Y0 | T=0]
    selection_bias = Y0[T == 1].mean() - Y0[T == 0].mean()
    
    # Naive estimate
    naive_diff = Y_obs[T == 1].mean() - Y_obs[T == 0].mean()
    
    print(f"\n" + "="*50)
    print("GENERATED OUTCOMES")
    print("="*50)
    print(f"\nTrue causal effects:")
    print(f"  ATT (target):     ${true_ATT:,.0f}")
    print(f"  ATE:              ${true_ATE:,.0f}")
    print(f"  ATU:              ${true_ATU:,.0f}")
    print(f"\nBias decomposition:")
    print(f"  Naive difference: ${naive_diff:,.0f}")
    print(f"  Selection bias:   ${selection_bias:,.0f}")
    print(f"  (Naive = ATT + Selection Bias)")
    print(f"\nBaseline Y(0) by group:")
    print(f"  E[Y(0) | T=1] (treated):   ${Y0[T == 1].mean():,.0f}")
    print(f"  E[Y(0) | T=0] (control):   ${Y0[T == 0].mean():,.0f}")
    print(f"  Gap (= selection bias):    ${selection_bias:,.0f}")
    
    # =========================================================================
    # SAVE DATA
    # =========================================================================
    
    df_clean['Y0'] = Y0
    df_clean['Y1'] = Y1
    df_clean['true_effect'] = tau
    df_clean['total_exp_semi'] = Y_obs
    df_clean['treatment_semi'] = T
    
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Save metadata
    metadata = {
        'true_effects': {
            'ATT': float(true_ATT), 
            'ATE': float(true_ATE),
            'ATU': float(true_ATU),
            'selection_bias': float(selection_bias),
            'naive_difference': float(naive_diff),
        },
        'generation_params': {
            'outcome_type': outcome_type,
            'base_effect': base_effect,
            'heterogeneous': heterogeneous,
            'age_confounding': age_confounding,
            'age_weight': age_weight,
            'other_weight': other_weight,
            'noise_std': noise_std,
            'seed': seed,
            'n_covariates': len(available),
            'covariates': available
        },
        'baseline_outcomes': {
            'E_Y0_treated': float(Y0[T == 1].mean()),
            'E_Y0_control': float(Y0[T == 0].mean()),
        }
    }
    
    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    return metadata

# def create_semisynthetic_meps(
#     processed_path: str = "meps_data/processed/meps_medicare_att.csv",
#     output_path: str = None,
#     outcome_type: str = 'linear',
#     base_effect: float = 2000.0,
#     heterogeneous: bool = True,
#     seed: int = 42,
# ):
#     """Create semi-synthetic MEPS data with known ground truth ATT."""
#     np.random.seed(seed)
    
#     if output_path is None:
#         output_path = processed_path.replace('.csv', '_semisynthetic.csv')
    
#     print("="*70)
#     print("CREATING SEMI-SYNTHETIC MEPS DATA")
#     print("="*70)
    
#     df = pd.read_csv(processed_path)
#     print(f"Loaded {len(df):,} observations")
    
#     confounder_vars = ['age', 'pov_level', 'pcs42', 'mcs42', 'phq2',
#                        'office_visits', 'outpatient_visits', 'er_visits',
#                        'inpatient_stays', 'rx_total', 'health_status', 'chronic_conditions']
#     available = [v for v in confounder_vars if v in df.columns]
    
#     df_clean = df.dropna(subset=available + ['treatment', 'total_exp']).copy()
    
#     X = df_clean[available].values
#     T = df_clean['treatment'].values
#     n = len(df_clean)
    
#     print(f"After dropping missing: {n:,} observations")
    
#     X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
#     if outcome_type == 'linear':
#         beta = np.array([200, 5, -50, -30, 100, 100, 150, 500, 1000, 50, -200, 300])[:len(available)]
#         Y0 = 5000 + X_std @ beta + np.random.normal(0, 2000, n)
#     else:
#         age_idx = available.index('age')
#         Y0 = 5000 + 200*X_std[:, age_idx]**2 + 100*X_std.sum(axis=1) + np.random.normal(0, 2000, n)
    
#     Y0 = np.clip(Y0, 0, None)
    
#     if heterogeneous:
#         age_idx = available.index('age')
#         tau = base_effect + 500 * X_std[:, age_idx]
#     else:
#         tau = np.full(n, base_effect)
    
#     Y1 = Y0 + tau
#     Y1 = np.clip(Y1, 0, None)
#     Y_obs = np.where(T == 1, Y1, Y0)
    
#     true_ATT = tau[T == 1].mean()
#     true_ATE = tau.mean()
    
#     print(f"\nGenerated outcomes:")
#     print(f"  True ATT: ${true_ATT:,.0f}")
#     print(f"  True ATE: ${true_ATE:,.0f}")
    
#     df_clean['Y0'] = Y0
#     df_clean['Y1'] = Y1
#     df_clean['true_effect'] = tau
#     df_clean['total_exp_semi'] = Y_obs
#     df_clean['treatment_semi'] = T
    
#     df_clean.to_csv(output_path, index=False)
#     print(f"\n✓ Saved to: {output_path}")
    
#     metadata = {
#         'true_effects': {'ATT': float(true_ATT), 'ATE': float(true_ATE)},
#         'generation_params': {
#             'outcome_type': outcome_type,
#             'base_effect': base_effect,
#             'heterogeneous': heterogeneous,
#             'seed': seed,
#             'n_covariates': len(available),
#             'covariates': available
#         }
#     }
    
#     metadata_path = output_path.replace('.csv', '_metadata.json')
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=2)
#     print(f"✓ Saved metadata to: {metadata_path}")
    
#     return metadata


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true', help='Process raw data')
    parser.add_argument('--confounding-level', type=str, default='severe')
    parser.add_argument('--year', type=int, default=2022)
    parser.add_argument('--create-semisynthetic', action='store_true')
    parser.add_argument('--data-dir', type=str, default='meps_data')
    parser.add_argument('--output-dir', type=str, default='meps_data/processed')
    args = parser.parse_args()
    
    if args.create_semisynthetic:
        processed_path= f'{args.output_dir}/meps_medicare_att.csv'
        semisynthetic_path = processed_path.replace('.csv', f'_semisynthetic_{args.confounding_level}.csv')
        # create_semisynthetic_meps(
        #     processed_path=processed_path,
        #     age_confounding=args.confounding_level
        # )
        data = load_meps_data(semisynthetic=True, semisynthetic_path=semisynthetic_path)
    else:
        data = load_meps_data(
            raw_data_dir=args.data_dir,
            processed_path=f'{args.output_dir}/meps_medicare_att.csv',
            process=args.process, 
            year=args.year
        )
    
    print(f"\n✓ Loaded data with {data['n0']} control, {data['n1']} treated, {data['d']} covariates")
    print(f"  Covariates: {data['covariate_names']}")

    print("\n" + "="*70)
    print("OVERLAP VISUALIZATION")
    print("="*70)
    
    # Create output directory for figures
    fig_dir = Path(args.output_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # Plot covariate distributions
    print("\n  Plotting covariate distributions...")
    plot_overlap(data, save_path=fig_dir / 'covariate_overlap.png', show=False)
    
    # Plot SMD summary
    print("  Plotting SMD summary...")
    fig, smds = plot_smd_summary(data, save_path=fig_dir / 'smd_summary.png', show=False)
    
    # Plot propensity scores
    print("  Plotting propensity score distributions...")
    plot_propensity_overlap(data, save_path=fig_dir / 'propensity_overlap.png', show=False)
    
    print(f"\n  All figures saved to: {fig_dir}/")
    
    # Print SMD summary
    print("\n  SMD Summary:")
    print("  " + "-"*40)
    for var, smd in sorted(smds.items(), key=lambda x: abs(x[1]), reverse=True):
        status = "⚠️ EXTREME" if abs(smd) > 1 else "⚠️ serious" if abs(smd) > 0.25 else "mild" if abs(smd) > 0.1 else "✓ ok"
        print(f"    {var:<25} {smd:>8.3f}  {status}")






