import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import json
from .gen_data import standardize_data


def load_semisynthetic_metadata(path):
    """Load metadata from semi-synthetic data generation."""
    metadata_path = path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'r') as f:
        return json.load(f)

def create_semisynthetic_rhc(
    original_path="data/rhc_processed.csv",
    save_path="data/rhc_semisynthetic.csv",
    save_metadata=True,
    true_att=2.0,
    overlap_strength='moderate',
    propensity_type='nonlinear',
    outcome_type='heterogeneous',
    seed=42,
    include_propensity_scores=True,
    include_counterfactuals=True,
    outcome_var='surv2md1',
    extreme_confounding=False,  # NEW: create severe non-overlap
    deterministic_fraction=0.1   # NEW: fraction with e(X) at 0 or 1
):
    """
    Create semi-synthetic RHC data with controllable overlap.
    
    Parameters:
    -----------
    extreme_confounding : bool
        If True, create severe non-overlap with some deterministic assignments
    deterministic_fraction : float
        Fraction of samples to push to deterministic treatment (e(X) = 0 or 1)
        Default 0.1 means ~10% will have propensity scores at extremes
    
    overlap_strength can now be:
    - 'good': ~80-90% overlap
    - 'moderate': ~60-70% overlap  
    - 'poor': ~40-50% overlap
    - 'severe': ~20-30% overlap
    - 'extreme': ~5-15% overlap with deterministic regions
    - float > 5.0: custom extreme confounding
    """
    np.random.seed(seed)
    
    # Load original data
    df = pd.read_csv(original_path)
    print(f"Loaded original RHC data: {df.shape}")
    
    # Define interpretable covariates
    severity_vars = ['aps1', 'scoma1']
    vital_vars = ['meanbp1', 'hrt1', 'resp1', 'temp1']
    lab_vars = ['crea1', 'alb1', 'hema1', 'pafi1']
    demo_vars = ['age']
    
    propensity_vars = severity_vars + vital_vars + lab_vars + demo_vars
    available_vars = [v for v in propensity_vars if v in df.columns]
    propensity_vars = available_vars
    
    covariate_cols = [col for col in df.columns if col not in ['swang1', 'dth30', 'surv2md1']]
    X_all = df[covariate_cols].values
    X_prop = df[propensity_vars].values
    n, d_prop = X_prop.shape
    
    print(f"\nCreating semi-synthetic RHC data...")
    print(f"  Sample size: {n}")
    print(f"  Outcome: {outcome_var}")
    print(f"  Extreme confounding: {extreme_confounding}")
    
    # Standardize propensity covariates
    X_prop_std = (X_prop - X_prop.mean(axis=0)) / X_prop.std(axis=0)
    
    # ========================================
    # Set overlap strength
    # ========================================
    overlap_map = {
        'good': 0.3,
        'moderate': 0.6,
        'poor': 1.0,
        'severe': 1.5,
        'extreme': 3.0  # NEW: very extreme
    }
    
    if isinstance(overlap_strength, str):
        overlap_coef = overlap_map.get(overlap_strength, 1.0)
        overlap_label = overlap_strength
        if overlap_strength == 'extreme':
            extreme_confounding = True
    else:
        overlap_coef = float(overlap_strength)
        overlap_label = f"custom ({overlap_strength:.2f})"
        if overlap_coef >= 5.0:
            extreme_confounding = True
    
    print(f"  Overlap strength: {overlap_label} (coef={overlap_coef:.2f})")
    
    # ========================================
    # Create EXTREME propensity scores
    # ========================================
    
    # Get raw covariate values for thresholding
    X_prop_raw = X_prop
    
    # Base coefficients - make them MUCH larger
    beta = np.zeros(d_prop)
    
    if 'aps1' in propensity_vars:
        idx = propensity_vars.index('aps1')
        beta[idx] = 1.5 * overlap_coef  # APACHE: very strong
    if 'scoma1' in propensity_vars:
        idx = propensity_vars.index('scoma1')
        beta[idx] = -1.0 * overlap_coef
    if 'meanbp1' in propensity_vars:
        idx = propensity_vars.index('meanbp1')
        beta[idx] = -0.8 * overlap_coef
    if 'hrt1' in propensity_vars:
        idx = propensity_vars.index('hrt1')
        beta[idx] = 0.6 * overlap_coef
    if 'crea1' in propensity_vars:
        idx = propensity_vars.index('crea1')
        beta[idx] = 0.8 * overlap_coef
    if 'age' in propensity_vars:
        idx = propensity_vars.index('age')
        beta[idx] = 0.5 * overlap_coef
    
    # Linear component
    logit_p = -0.5 + X_prop_std @ beta
    
    # Add strong nonlinear terms
    if propensity_type == 'nonlinear':
        gamma = np.zeros(d_prop)
        if 'aps1' in propensity_vars:
            idx = propensity_vars.index('aps1')
            # Very sick patients → much more likely to get RHC
            gamma[idx] = 0.5 * overlap_coef
        if 'age' in propensity_vars:
            idx = propensity_vars.index('age')
            gamma[idx] = 0.3 * overlap_coef
        
        logit_p += (X_prop_std**2) @ gamma
        
        # Add multiple strong interactions
        if 'aps1' in propensity_vars and 'meanbp1' in propensity_vars:
            idx_aps = propensity_vars.index('aps1')
            idx_bp = propensity_vars.index('meanbp1')
            # Sick + low BP → MUCH more RHC
            logit_p += 0.5 * overlap_coef * X_prop_std[:, idx_aps] * X_prop_std[:, idx_bp]
        
        if 'age' in propensity_vars and 'crea1' in propensity_vars:
            idx_age = propensity_vars.index('age')
            idx_crea = propensity_vars.index('crea1')
            # Old + kidney failure → more RHC
            logit_p += 0.4 * overlap_coef * X_prop_std[:, idx_age] * X_prop_std[:, idx_crea]
    
    # ========================================
    # Add EXTREME CONFOUNDING regions
    # ========================================
    if extreme_confounding:
        print(f"  Creating extreme confounding regions...")
        
        # Create deterministic treatment regions based on clinical logic
        # Very sick patients (top 5-10%) → ALWAYS get RHC (e(X) → 1)
        # Very healthy patients (bottom 5-10%) → NEVER get RHC (e(X) → 0)
        
        if 'aps1' in propensity_vars:
            idx_aps = propensity_vars.index('aps1')
            aps_values = X_prop_raw[:, idx_aps]
            aps_percentile_high = np.percentile(aps_values, 100 - deterministic_fraction * 50)
            aps_percentile_low = np.percentile(aps_values, deterministic_fraction * 50)
            
            # Very sick: push logit to +∞
            very_sick = aps_values > aps_percentile_high
            logit_p[very_sick] += 10.0  # Push to e(X) ≈ 1
            
            # Very healthy: push logit to -∞
            very_healthy = aps_values < aps_percentile_low
            logit_p[very_healthy] -= 10.0  # Push to e(X) ≈ 0
            
            print(f"    Very sick patients (APACHE > {aps_percentile_high:.1f}): {very_sick.sum()} → e(X) ≈ 1")
            print(f"    Very healthy patients (APACHE < {aps_percentile_low:.1f}): {very_healthy.sum()} → e(X) ≈ 0")
        
        # Additional extreme region: combine multiple risk factors
        if 'age' in propensity_vars and 'meanbp1' in propensity_vars:
            idx_age = propensity_vars.index('age')
            idx_bp = propensity_vars.index('meanbp1')
            
            age_values = X_prop_raw[:, idx_age]
            bp_values = X_prop_raw[:, idx_bp]
            
            # Old + low BP + not already in extreme regions → high propensity
            age_high = np.percentile(age_values, 80)
            bp_low = np.percentile(bp_values, 20)
            
            high_risk = (age_values > age_high) & (bp_values < bp_low)
            if 'aps1' in propensity_vars:
                # Don't double-count those already in extreme regions
                high_risk = high_risk & ~very_sick & ~very_healthy
            
            logit_p[high_risk] += 8.0
            print(f"    High-risk combo (old + low BP): {high_risk.sum()} → e(X) ≈ 0.99")
    
    # Convert to probabilities
    propensity_scores = 1 / (1 + np.exp(-logit_p))
    
    # For extreme confounding, clip less aggressively to preserve extremes
    if extreme_confounding:
        propensity_scores = np.clip(propensity_scores, 0.001, 0.999)
    else:
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
    
    # ========================================
    # Diagnose propensity scores
    # ========================================
    n_near_zero = (propensity_scores < 0.05).sum()
    n_near_one = (propensity_scores > 0.95).sum()
    n_extreme = n_near_zero + n_near_one
    
    print(f"\n  Propensity score diagnostics:")
    print(f"    Range: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
    print(f"    Mean: {propensity_scores.mean():.3f}")
    print(f"    Near 0 (< 0.05): {n_near_zero} ({n_near_zero/n*100:.1f}%)")
    print(f"    Near 1 (> 0.95): {n_near_one} ({n_near_one/n*100:.1f}%)")
    print(f"    Total extreme: {n_extreme} ({n_extreme/n*100:.1f}%)")
    
    # ========================================
    # Assign treatment
    # ========================================
    T_new = np.random.binomial(1, propensity_scores)
    
    # ========================================
    # Generate outcomes with STRONG confounding
    # ========================================
    
    log_baseline = 2.0
    
    # Use SAME variables with LARGE coefficients (creates confounding)
    confounding_strength = 1.0 + overlap_coef * 0.3  # Scale confounding with overlap
    
    if 'aps1' in propensity_vars:
        idx = propensity_vars.index('aps1')
        log_baseline += -0.6 * confounding_strength * X_prop_std[:, idx]
    
    if 'scoma1' in propensity_vars:
        idx = propensity_vars.index('scoma1')
        log_baseline += 0.4 * confounding_strength * X_prop_std[:, idx]
    
    if 'meanbp1' in propensity_vars:
        idx = propensity_vars.index('meanbp1')
        log_baseline += 0.35 * confounding_strength * X_prop_std[:, idx]
    
    if 'age' in propensity_vars:
        idx = propensity_vars.index('age')
        log_baseline += -0.3 * confounding_strength * X_prop_std[:, idx]
    
    if 'crea1' in propensity_vars:
        idx = propensity_vars.index('crea1')
        log_baseline += -0.3 * confounding_strength * X_prop_std[:, idx]
    
    # Add nonlinear confounding
    if outcome_type in ['heterogeneous', 'nonlinear']:
        if 'aps1' in propensity_vars:
            idx = propensity_vars.index('aps1')
            log_baseline += -0.2 * confounding_strength * (X_prop_std[:, idx]**2)
        
        if 'age' in propensity_vars and 'aps1' in propensity_vars:
            idx_age = propensity_vars.index('age')
            idx_aps = propensity_vars.index('aps1')
            log_baseline += -0.25 * confounding_strength * X_prop_std[:, idx_age] * X_prop_std[:, idx_aps]
    
    noise_std = 0.8
    log_Y0 = log_baseline + np.random.normal(0, noise_std, n)
    Y0 = np.exp(log_Y0)
    Y0 = np.clip(Y0, 0.1, 100)
    
    # Treatment effect
    if outcome_type == 'constant':
        treatment_effect = np.full(n, true_att)
        
    elif outcome_type == 'heterogeneous':
        effect_modifier = np.ones(n)
        
        if 'aps1' in propensity_vars:
            idx = propensity_vars.index('aps1')
            effect_modifier += 0.6 * X_prop_std[:, idx]
        
        if 'age' in propensity_vars:
            idx = propensity_vars.index('age')
            effect_modifier += -0.3 * X_prop_std[:, idx]
        
        effect_modifier = effect_modifier / effect_modifier.mean()
        treatment_effect = true_att * effect_modifier
        
    elif outcome_type == 'nonlinear':
        effect_modifier = np.ones(n)
        
        if 'aps1' in propensity_vars:
            idx = propensity_vars.index('aps1')
            effect_modifier += 0.7 * X_prop_std[:, idx] - 0.25 * (X_prop_std[:, idx]**2)
        
        if 'age' in propensity_vars and 'meanbp1' in propensity_vars:
            idx_age = propensity_vars.index('age')
            idx_bp = propensity_vars.index('meanbp1')
            effect_modifier += 0.2 * X_prop_std[:, idx_age] * X_prop_std[:, idx_bp]
        
        effect_modifier = effect_modifier / effect_modifier.mean()
        treatment_effect = true_att * effect_modifier
    
    Y1 = Y0 + treatment_effect
    Y1 = np.clip(Y1, 0.1, 100)
    
    Y_new = T_new * Y1 + (1 - T_new) * Y0
    
    # ========================================
    # Compute true effects and overlap
    # ========================================
    treated_mask = (T_new == 1)
    control_mask = ~treated_mask
    
    true_att_actual = (Y1[treated_mask] - Y0[treated_mask]).mean()
    true_ate = (Y1 - Y0).mean()
    true_atc = (Y1[control_mask] - Y0[control_mask]).mean()
    
    ps_treated = propensity_scores[treated_mask]
    ps_control = propensity_scores[control_mask]
    
    common_support_lower = max(ps_treated.min(), ps_control.min())
    common_support_upper = min(ps_treated.max(), ps_control.max())
    
    in_common_treated = ((ps_treated >= common_support_lower) & 
                         (ps_treated <= common_support_upper)).mean()
    in_common_control = ((ps_control >= common_support_lower) & 
                         (ps_control <= common_support_upper)).mean()
    
    # ========================================
    # Save data
    # ========================================
    df_semi = df.copy()
    df_semi['swang1_semi'] = T_new
    df_semi[f'{outcome_var}_semi'] = Y_new
    
    if include_propensity_scores:
        df_semi['propensity_score'] = propensity_scores
    
    if include_counterfactuals:
        df_semi['Y0'] = Y0
        df_semi['Y1'] = Y1
        df_semi['true_effect'] = Y1 - Y0
    
    df_semi.to_csv(save_path, index=False)
    print(f"\n✓ Saved semi-synthetic data to: {save_path}")
    
    # ========================================
    # Create metadata
    # ========================================
    metadata = {
        'file_path': save_path,
        'creation_date': pd.Timestamp.now().isoformat(),
        'original_data': original_path,
        'outcome_variable': outcome_var,
        'outcome_type': 'continuous',
        'sample_size': int(n),
        'n_treated': int(treated_mask.sum()),
        'n_control': int(control_mask.sum()),
        'n_covariates': int(X_all.shape[1]),
        'propensity_covariates': propensity_vars,
        'generation_params': {
            'true_att_target': true_att,
            'overlap_strength': overlap_label,
            'overlap_coefficient': float(overlap_coef),
            'propensity_type': propensity_type,
            'outcome_type': outcome_type,
            'extreme_confounding': extreme_confounding,
            'deterministic_fraction': deterministic_fraction if extreme_confounding else 0.0,
            'seed': seed
        },
        'true_effects': {
            'ATT': float(true_att_actual),
            'ATE': float(true_ate),
            'ATC': float(true_atc)
        },
        'observed_effects': {
            'treated_mean': float(Y_new[treated_mask].mean()),
            'control_mean': float(Y_new[control_mask].mean()),
            'naive_difference': float(Y_new[treated_mask].mean() - Y_new[control_mask].mean()),
            'selection_bias': float((Y_new[treated_mask].mean() - Y_new[control_mask].mean()) - true_att_actual)
        },
        'propensity_diagnostics': {
            'treated_mean': float(ps_treated.mean()),
            'control_mean': float(ps_control.mean()),
            'treated_range': [float(ps_treated.min()), float(ps_treated.max())],
            'control_range': [float(ps_control.min()), float(ps_control.max())],
            'common_support': [float(common_support_lower), float(common_support_upper)],
            'fraction_in_common_support': {
                'treated': float(in_common_treated),
                'control': float(in_common_control),
                'average': float((in_common_treated + in_common_control) / 2)
            },
            'n_extreme': int(n_extreme),
            'n_near_zero': int(n_near_zero),
            'n_near_one': int(n_near_one),
            'pct_extreme': float(n_extreme / n * 100)
        }
    }
    
    # ========================================
    # Print summary
    # ========================================
    print(f"\n{'='*70}")
    print("SEMI-SYNTHETIC DATA SUMMARY")
    print('='*70)
    print(f"  Sample: Control={control_mask.sum()}, Treated={treated_mask.sum()}")
    print(f"\n  TRUE CAUSAL EFFECTS:")
    print(f"    ATT: {true_att_actual:.2f} months (target: {true_att:.2f})")
    print(f"    ATE: {true_ate:.2f} months")
    print(f"\n  OBSERVED (CONFOUNDED):")
    print(f"    Treated survival: {Y_new[treated_mask].mean():.2f} months")
    print(f"    Control survival: {Y_new[control_mask].mean():.2f} months")
    print(f"    Naive difference: {Y_new[treated_mask].mean() - Y_new[control_mask].mean():.2f} months")
    print(f"    → Selection bias: {metadata['observed_effects']['selection_bias']:.2f} months")
    print(f"\n  OVERLAP QUALITY:")
    print(f"    Treated e(X): {ps_treated.mean():.3f} [{ps_treated.min():.4f}, {ps_treated.max():.4f}]")
    print(f"    Control e(X): {ps_control.mean():.3f} [{ps_control.min():.4f}, {ps_control.max():.4f}]")
    print(f"    Common support: [{common_support_lower:.4f}, {common_support_upper:.4f}]")
    print(f"    In common support: {metadata['propensity_diagnostics']['fraction_in_common_support']['average']:.1%}")
    print(f"    Extreme e(X) (< 0.05 or > 0.95): {n_extreme} ({n_extreme/n*100:.1f}%)")
    
    overlap_avg = metadata['propensity_diagnostics']['fraction_in_common_support']['average']
    if overlap_avg > 0.75:
        difficulty = "EASY - IPW should work well"
    elif overlap_avg > 0.55:
        difficulty = "MODERATE - IPW may struggle"
    elif overlap_avg > 0.35:
        difficulty = "HARD - IPW will struggle significantly"
    elif overlap_avg > 0.15:
        difficulty = "VERY HARD - IPW likely to fail"
    else:
        difficulty = "EXTREME - IPW will almost certainly fail"
    
    print(f"    → Difficulty: {difficulty}")
    print('='*70)
    
    # Save metadata
    if save_metadata:
        import json
        metadata_path = save_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✓ Saved metadata to: {metadata_path}")
    
    return metadata

# def create_semisynthetic_rhc(
#     original_path="data/rhc_processed.csv",
#     save_path="data/rhc_semisynthetic.csv",
#     save_metadata=True,
#     true_att=2.0,
#     overlap_strength='moderate',
#     propensity_type='nonlinear',
#     outcome_type='heterogeneous',
#     seed=42,
#     include_propensity_scores=True,
#     include_counterfactuals=True,
#     outcome_var='surv2md1',
#     confounding_strength=1.0  # NEW: control confounding magnitude
# ):
#     """
#     Create semi-synthetic RHC data with stronger, more complex confounding.
    
#     Parameters:
#     -----------
#     confounding_strength : float
#         Multiplier for confounding (1.0 = moderate, 2.0 = strong)
#         Higher values create stronger selection bias
#     """
#     np.random.seed(seed)
    
#     # Load and setup (same as before)
#     df = pd.read_csv(original_path)
    
#     severity_vars = ['aps1', 'scoma1']
#     vital_vars = ['meanbp1', 'hrt1', 'resp1', 'temp1']
#     lab_vars = ['crea1', 'alb1', 'hema1', 'pafi1']
#     demo_vars = ['age']
    
#     propensity_vars = severity_vars + vital_vars + lab_vars + demo_vars
#     available_vars = [v for v in propensity_vars if v in df.columns]
#     propensity_vars = available_vars
    
#     covariate_cols = [col for col in df.columns if col not in ['swang1', 'dth30', 'surv2md1']]
#     X_all = df[covariate_cols].values
#     X_prop = df[propensity_vars].values
#     n, d_prop = X_prop.shape
    
#     print(f"\nCreating semi-synthetic RHC data...")
#     print(f"  Confounding strength: {confounding_strength}")
    
#     # Standardize
#     X_prop_std = (X_prop - X_prop.mean(axis=0)) / X_prop.std(axis=0)
    
#     # Overlap coefficient
#     overlap_map = {'good': 0.3, 'moderate': 0.6, 'poor': 1.0, 'severe': 1.5}
#     if isinstance(overlap_strength, str):
#         overlap_coef = overlap_map[overlap_strength]
#         overlap_label = overlap_strength
#     else:
#         overlap_coef = 0.1 + overlap_strength * 1.4
#         overlap_label = f"custom ({overlap_strength:.2f})"
    
#     # ========================================
#     # PROPENSITY SCORE - More complex
#     # ========================================
#     beta = np.zeros(d_prop)
    
#     if 'aps1' in propensity_vars:
#         idx = propensity_vars.index('aps1')
#         beta[idx] = 0.8 * overlap_coef
#     if 'scoma1' in propensity_vars:
#         idx = propensity_vars.index('scoma1')
#         beta[idx] = -0.5 * overlap_coef
#     if 'meanbp1' in propensity_vars:
#         idx = propensity_vars.index('meanbp1')
#         beta[idx] = -0.4 * overlap_coef
#     if 'hrt1' in propensity_vars:
#         idx = propensity_vars.index('hrt1')
#         beta[idx] = 0.3 * overlap_coef
#     if 'resp1' in propensity_vars:
#         idx = propensity_vars.index('resp1')
#         beta[idx] = 0.3 * overlap_coef
#     if 'crea1' in propensity_vars:
#         idx = propensity_vars.index('crea1')
#         beta[idx] = 0.4 * overlap_coef
#     if 'age' in propensity_vars:
#         idx = propensity_vars.index('age')
#         beta[idx] = 0.2 * overlap_coef
    
#     logit_p = -0.5 + X_prop_std @ beta
    
#     # Add complex nonlinear terms
#     if propensity_type == 'nonlinear':
#         # Quadratic terms
#         gamma = np.zeros(d_prop)
#         if 'aps1' in propensity_vars:
#             idx = propensity_vars.index('aps1')
#             gamma[idx] = 0.2 * overlap_coef
#         if 'age' in propensity_vars:
#             idx = propensity_vars.index('age')
#             gamma[idx] = 0.15 * overlap_coef
        
#         logit_p += (X_prop_std**2) @ gamma
        
#         # Multiple interactions
#         if 'aps1' in propensity_vars and 'meanbp1' in propensity_vars:
#             idx_aps = propensity_vars.index('aps1')
#             idx_bp = propensity_vars.index('meanbp1')
#             logit_p += 0.25 * overlap_coef * X_prop_std[:, idx_aps] * X_prop_std[:, idx_bp]
        
#         if 'age' in propensity_vars and 'crea1' in propensity_vars:
#             idx_age = propensity_vars.index('age')
#             idx_crea = propensity_vars.index('crea1')
#             logit_p += 0.2 * overlap_coef * X_prop_std[:, idx_age] * X_prop_std[:, idx_crea]
    
#     propensity_scores = 1 / (1 + np.exp(-logit_p))
#     propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
    
#     T_new = np.random.binomial(1, propensity_scores)
    
#     # ========================================
#     # OUTCOME MODEL - More complex confounding
#     # ========================================
    
#     # Key insight: Use THE SAME covariates that predict treatment
#     # This creates confounding!
    
#     log_baseline = 2.0  # Base survival
    
#     # Add confounders - same variables that predict treatment
#     # This creates the selection bias!
#     if 'aps1' in propensity_vars:
#         idx = propensity_vars.index('aps1')
#         # CRITICAL: Sicker patients (high APACHE) get RHC AND have worse survival
#         log_baseline += -0.5 * confounding_strength * X_prop_std[:, idx]
    
#     if 'scoma1' in propensity_vars:
#         idx = propensity_vars.index('scoma1')
#         log_baseline += 0.35 * confounding_strength * X_prop_std[:, idx]
    
#     if 'meanbp1' in propensity_vars:
#         idx = propensity_vars.index('meanbp1')
#         # Low BP → more likely to get RHC AND worse survival
#         log_baseline += 0.3 * confounding_strength * X_prop_std[:, idx]
    
#     if 'age' in propensity_vars:
#         idx = propensity_vars.index('age')
#         # Older → more likely to get RHC AND worse survival
#         log_baseline += -0.25 * confounding_strength * X_prop_std[:, idx]
    
#     if 'crea1' in propensity_vars:
#         idx = propensity_vars.index('crea1')
#         log_baseline += -0.25 * confounding_strength * X_prop_std[:, idx]
    
#     # Add nonlinear confounding
#     if outcome_type in ['heterogeneous', 'nonlinear']:
#         if 'aps1' in propensity_vars:
#             idx = propensity_vars.index('aps1')
#             # Quadratic effect - very sick patients have much worse outcomes
#             log_baseline += -0.15 * confounding_strength * (X_prop_std[:, idx]**2)
        
#         # Interaction: older + sicker = much worse
#         if 'age' in propensity_vars and 'aps1' in propensity_vars:
#             idx_age = propensity_vars.index('age')
#             idx_aps = propensity_vars.index('aps1')
#             log_baseline += -0.2 * confounding_strength * X_prop_std[:, idx_age] * X_prop_std[:, idx_aps]
    
#     # Generate baseline outcomes
#     noise_std = 0.8
#     log_Y0 = log_baseline + np.random.normal(0, noise_std, n)
#     Y0 = np.exp(log_Y0)
#     Y0 = np.clip(Y0, 0.1, 100)
    
#     # Treatment effect (heterogeneous)
#     if outcome_type == 'constant':
#         treatment_effect = np.full(n, true_att)
        
#     elif outcome_type == 'heterogeneous':
#         effect_modifier = np.ones(n)
        
#         if 'aps1' in propensity_vars:
#             idx = propensity_vars.index('aps1')
#             # More benefit for sicker patients
#             effect_modifier += 0.6 * X_prop_std[:, idx]
        
#         if 'age' in propensity_vars:
#             idx = propensity_vars.index('age')
#             # Less benefit for very old
#             effect_modifier += -0.3 * X_prop_std[:, idx]
        
#         effect_modifier = effect_modifier / effect_modifier.mean()
#         treatment_effect = true_att * effect_modifier
        
#     elif outcome_type == 'nonlinear':
#         effect_modifier = np.ones(n)
        
#         if 'aps1' in propensity_vars:
#             idx = propensity_vars.index('aps1')
#             # Nonlinear: max benefit for moderately sick
#             effect_modifier += 0.7 * X_prop_std[:, idx] - 0.25 * (X_prop_std[:, idx]**2)
        
#         if 'age' in propensity_vars and 'meanbp1' in propensity_vars:
#             idx_age = propensity_vars.index('age')
#             idx_bp = propensity_vars.index('meanbp1')
#             # Interaction in treatment effect
#             effect_modifier += 0.2 * X_prop_std[:, idx_age] * X_prop_std[:, idx_bp]
        
#         effect_modifier = effect_modifier / effect_modifier.mean()
#         treatment_effect = true_att * effect_modifier
    
#     Y1 = Y0 + treatment_effectlinear

#     Y1 = np.clip(Y1, 0.1, 100)
    
#     Y_new = T_new * Y1 + (1 - T_new) * Y0
    
#     # Rest of the function (compute effects, save, etc.) - same as before
#     treated_mask = (T_new == 1)
#     control_mask = ~treated_mask
    
#     true_att_actual = (Y1[treated_mask] - Y0[treated_mask]).mean()
#     true_ate = (Y1 - Y0).mean()
    
#     # Save propensity scores with full array
#     ps_treated = propensity_scores[treated_mask]
#     ps_control = propensity_scores[control_mask]
    
#     # [Rest of metadata and saving code same as before...]
    
#     df_semi = df.copy()
#     df_semi['swang1_semi'] = T_new
#     df_semi[f'{outcome_var}_semi'] = Y_new
    
#     if include_propensity_scores:
#         df_semi['propensity_score'] = propensity_scores  # Save all propensity scores
    
#     if include_counterfactuals:
#         df_semi['Y0'] = Y0
#         df_semi['Y1'] = Y1
#         df_semi['true_effect'] = Y1 - Y0
    
#     df_semi.to_csv(save_path, index=False)
    
#     # Compute selection bias
#     selection_bias = (Y_new[treated_mask].mean() - Y_new[control_mask].mean()) - true_att_actual
    
#     print(f"\n{'='*70}")
#     print("SEMI-SYNTHETIC DATA SUMMARY")
#     print('='*70)
#     print(f"  TRUE ATT: {true_att_actual:.2f} months")
#     print(f"  OBSERVED DIFFERENCE: {Y_new[treated_mask].mean() - Y_new[control_mask].mean():.2f} months")
#     print(f"  SELECTION BIAS: {selection_bias:.2f} months")
#     print(f"    → Methods must correct bias of {abs(selection_bias):.2f} months!")
#     print('='*70)
    
#     metadata = {
#         'true_effects': {'ATT': float(true_att_actual), 'ATE': float(true_ate)},
#         'selection_bias': float(selection_bias),
#         'propensity_scores_all': propensity_scores.tolist(),  # Save for IPW
#         # ... other metadata
#     }
    
#     return metadata

def load_rhc_data(
    path="data/rhc_processed.csv", 
    download=False,
    standardize=True,
    standardize_method: str = 'zscore',
    semisynthetic=False,
    semisynthetic_params=None,
    semisynthetic_path=None,
    ):
    """
    Load Right Heart Catheterization (RHC) dataset and prepare for causal inference.
    """
    metadata = None
    
    # Default for original data
    treat_var = 'swang1'
    outcome_var = 'surv2md1'
    outcome_info = {
        'name': 'surv2md1',
        'description': 'Survival time in months',
        'type': 'positive'
    }

    continuous_vars = [
        'age', 'edu', 'das2d3pc', 'aps1', 'scoma1',
        'meanbp1', 'wblc1', 'hrt1', 'resp1', 'temp1', 'pafi1',
        'alb1', 'hema1', 'bili1', 'crea1', 'sod1', 'pot1', 
        'paco21', 'ph1', 'wtkilo1',
    ]

    if download:
        # [Your existing download code - unchanged]
        url = "https://hbiostat.org/data/repo/rhc.csv"
        df = pd.read_csv(url)
        print(f"Loaded RHC dataset: {df.shape[0]} patients, {df.shape[1]} variables")
        
        df['swang1'] = (df['swang1'] == 'RHC').astype(int).values
        
        available_vars = []
        for var in continuous_vars:
            if var in df.columns:
                missing_pct = df[var].isna().sum() / len(df) * 100
                if missing_pct < 50:
                    available_vars.append(var)
                    print(f"  {var}: {missing_pct:.1f}% missing")
                else:
                    print(f"  {var}: EXCLUDED ({missing_pct:.1f}% missing)")
            else:
                print(f"  {var}: NOT FOUND in dataset")
        
        prep_df = df[available_vars + ['swang1', 'surv2md1']].copy()
        
        for col in prep_df.columns:
            if prep_df[col].isna().any():
                prep_df[col] = prep_df[col].fillna(prep_df[col].mean())

        prep_df.to_csv(path, index=False)
        print(f"  Saved to: {path}")

    else:
        # Load existing data
        if semisynthetic:
            if semisynthetic_path is None:
                semisynthetic_path = path.replace('.csv', '_semisynthetic.csv')
            
            if not os.path.exists(semisynthetic_path):
                raise FileNotFoundError(
                    f"Semi-synthetic data not found at {semisynthetic_path}. "
                    f"Generate it first using create_semisynthetic_rhc()."
                )
            
            print(f"\nLoading SEMI-SYNTHETIC data from: {semisynthetic_path}")
            prep_df = pd.read_csv(semisynthetic_path)
            
            # Load metadata with robust error handling
            try:
                metadata_path = semisynthetic_path.replace('.csv', '_metadata.json')
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"Loaded metadata from: {metadata_path}")
                
                # Robust key access with defaults
                outcome_var_base = metadata.get('outcome_variable', 'surv2md1')
                outcome_type = metadata.get('outcome_type', 'continuous')
                
                # Handle nested dictionaries safely
                true_effects = metadata.get('true_effects', {})
                generation_params = metadata.get('generation_params', {})
                
                print(f"\n{'='*60}")
                print("SEMI-SYNTHETIC DATA INFO")
                print('='*60)
                print(f"  True ATT: {true_effects.get('ATT', 'N/A')}")
                print(f"  True ATE: {true_effects.get('ATE', 'N/A')}")
                print(f"  Outcome variable: {outcome_var_base} ({outcome_type})")
                
                if generation_params:
                    print(f"  Overlap strength: {generation_params.get('overlap_strength', 'N/A')}")
                    print(f"  Propensity type: {generation_params.get('propensity_type', 'N/A')}")
                    print(f"  Outcome type: {generation_params.get('outcome_type', 'N/A')}")
                
                print('='*60)
                
            except FileNotFoundError:
                print(f"  Warning: Metadata file not found at {metadata_path}")
                print(f"  Trying to infer from data columns...")
                metadata = None
                outcome_var_base = 'surv2md1'
                outcome_type = 'continuous'
                
            except Exception as e:
                print(f"  Warning: Error loading metadata: {e}")
                print(f"  Trying to infer from data columns...")
                metadata = None
                outcome_var_base = 'surv2md1'
                outcome_type = 'continuous'
            
            # Infer outcome from columns if metadata failed
            if metadata is None:
                if 'surv2md1_semi' in prep_df.columns:
                    outcome_var_base = 'surv2md1'
                    outcome_type = 'continuous'
                elif 'dth30_semi' in prep_df.columns:
                    outcome_var_base = 'dth30'
                    outcome_type = 'binary'
                else:
                    raise ValueError("Could not determine outcome variable from data!")
                
                print(f"  Inferred outcome: {outcome_var_base} ({outcome_type})")
            
            # Use semi-synthetic columns
            treat_var = 'swang1_semi'
            outcome_var = f'{outcome_var_base}_semi'
            
            outcome_info = {
                'name': outcome_var_base,
                'description': 'Survival time in months' if outcome_var_base == 'surv2md1' else '30-day mortality',
                'type': outcome_type
            }
            
            print(f"\n  Using columns:")
            print(f"    Treatment: {treat_var}")
            print(f"    Outcome: {outcome_var}")
            
            # Verify columns exist
            if treat_var not in prep_df.columns:
                raise ValueError(f"Treatment column '{treat_var}' not found in {list(prep_df.columns)}")
            if outcome_var not in prep_df.columns:
                raise ValueError(f"Outcome column '{outcome_var}' not found in {list(prep_df.columns)}")
            
        else:
            print(f"\nLoading ORIGINAL observational data from: {path}")
            prep_df = pd.read_csv(path)
            metadata = None
            treat_var = 'swang1'
            outcome_var = 'surv2md1'
            outcome_info = {
                'name': 'surv2md1',
                'description': 'Survival time in months',
                'type': 'continuous'
            }
    
    # Extract covariates (exclude ALL outcome/treatment columns)
    exclude_cols = [
        treat_var, outcome_var,
        'swang1', 'swang1_semi',
        'dth30', 'dth30_semi', 
        'surv2md1', 'surv2md1_semi',
        'propensity_score', 'Y0', 'Y1', 'true_effect'
    ]
    
    available_continuous_vars = [col for col in prep_df.columns if col not in exclude_cols]
    
    X = prep_df[available_continuous_vars].values
    T = prep_df[treat_var].values
    
    # Handle outcome based on type
    if outcome_info['type'] in ['continuous', 'positive']:
        Y = prep_df[outcome_var].values.astype(float)
    else:
        # For binary outcomes
        Y_raw = prep_df[outcome_var].values
        if Y_raw.dtype == object:
            Y = (Y_raw == "Yes").astype(float)
        else:
            Y = Y_raw.astype(float)
    
    print(f"\n  Extracted data:")
    print(f"    X shape: {X.shape}")
    print(f"    Covariates: {available_continuous_vars[:5]}{'...' if len(available_continuous_vars) > 5 else ''}")
    print(f"    T: {T.sum()} treated, {(T==0).sum()} control")
    print(f"    Y ({outcome_var}): range=[{Y.min():.2f}, {Y.max():.2f}], mean={Y.mean():.2f}")

    # Split by treatment status
    control_mask = (T == 0)
    treatment_mask = (T == 1)
    
    X_control = X[control_mask]
    X_treatment = X[treatment_mask]
    Y_control = Y[control_mask]
    Y_treatment = Y[treatment_mask]
    
    n0 = X_control.shape[0]
    n1 = X_treatment.shape[0]
    d = X.shape[1]

    print(f"\nFinal dataset:")
    print(f"  Control units: {n0}")
    print(f"  Treatment units: {n1}")
    print(f"  Covariates: {d}")
    
    print(f"\nObserved outcomes:")
    print(f"  Control: {Y_control.mean():.3f}")
    print(f"  Treatment: {Y_treatment.mean():.3f}")
    print(f"  Naive difference: {Y_treatment.mean() - Y_control.mean():.3f}")
    
    if metadata is not None and 'true_effects' in metadata:
        true_att = metadata['true_effects'].get('ATT', None)
        if true_att is not None:
            selection_bias = (Y_treatment.mean() - Y_control.mean()) - true_att
            print(f"  True ATT: {true_att:.3f}")
            print(f"  → Selection bias: {selection_bias:.3f}")
    
    # Create data dictionary
    data = {
        'X_control': X_control,
        'X_treatment': X_treatment,
        'Y_control': Y_control,
        'Y_treatment': Y_treatment,
        'covariate_names': available_continuous_vars,
        'outcome_info': outcome_info,
        'true_ATT': None,
        'Y0_treatment': None, 
        'n0': n0,
        'n1': n1,
        'd': d
    }
    
    # Add ground truth if available
    if metadata is not None:
        true_effects = metadata.get('true_effects', {})
        data['true_ATT'] = true_effects.get('ATT', None)
        data['true_ATE'] = true_effects.get('ATE', None)
        data['metadata'] = metadata
        
        # Load counterfactuals if available
        if 'Y0' in prep_df.columns:
            data['Y0_treatment'] = prep_df.loc[treatment_mask, 'Y0'].values
            data['Y1_treatment'] = prep_df.loc[treatment_mask, 'Y1'].values
            
            # Verify counterfactuals
            computed_att = (data['Y1_treatment'] - data['Y0_treatment']).mean()
            print(f"\n  ✓ Verification from counterfactuals: ATT = {computed_att:.3f}")
            
            if data['true_ATT'] is not None and abs(computed_att - data['true_ATT']) > 0.01:
                print(f"    Warning: Mismatch with metadata ATT ({data['true_ATT']:.3f})")
    
    # Standardize if requested
    if standardize:
        print(f"\nStandardizing using {standardize_method} method...")
        data, scaler_params = standardize_data(data, method=standardize_method)
        data['scaler_params'] = scaler_params
        
        X_std = np.vstack([data['X_control'], data['X_treatment']])
        print(f"  After standardization:")
        print(f"    Mean: {X_std.mean(axis=0).mean():.3f} (should be ~0)")
        print(f"    Std: {X_std.std(axis=0).mean():.3f} (should be ~1)")
    else:
        data['scaler_params'] = None
    
    return data


if __name__ == "__main__":
    # Builds the semi-synthetic RHC data used in the paper.
    # Run from the repository root: python -m src.data_utils.gen_rhc
    os.makedirs("data", exist_ok=True)

    # Step 1: download the original RHC data and keep the continuous covariates.
    if not os.path.exists("data/rhc_processed.csv"):
        load_rhc_data(
            path="data/rhc_processed.csv",
            download=True,
            standardize=True,
            )

    # Step 2: regenerate treatment with near-deterministic propensities (severe non-overlap).
    metadata = create_semisynthetic_rhc(
        original_path="data/rhc_processed.csv",
        save_path="data/rhc_nightmare.csv",
        true_att=3.0,
        overlap_strength=5.0,  # Very large coefficient
        extreme_confounding=True,
        deterministic_fraction=0.4,
        outcome_type="nonlinear",
        seed=42
    )
