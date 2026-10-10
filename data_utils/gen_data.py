import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import json

def gen_beta_data(rng=np.random.default_rng(0), n0=300, n1=300, d=2, 
                 overlap_param=3.0, outcome_type='linear', 
                 sigma=0.01, plot=False):
    rng_beta = np.random.default_rng(0)
    n = n1 + n0
    beta1 = np.array([-1.1] + list(rng_beta.normal(size=d)+1))
    beta0 = np.array([1.2] +  list(rng_beta.normal(size=d)+1))
    # (n, r, beta0, beta1, s=0, sigma0=.01, linear=False, plot=False):
    a1, b1 = 1.3, 1.55 - 0.1 * overlap_param
    a0, b0 = 1.55-0.1*overlap_param, 2.

    X1 = rng.beta(a1, b1, size=(n1,d)) * 50 - 25 #+ np.random.normal(size=(n1,d)) * sigma0
    X0 = rng.beta(a0, b0, size=(n0,d)) * 50 - 25 #+ np.random.normal(size=(n0,d)) * sigma0

    if plot:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(X1[:,0], label=r'$P_X^1$', fill=True, alpha=0.5)
        sns.kdeplot(X0[:,0], label=r'$P_X^0$', fill=True, alpha=0.5)

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        plt.close()

    X_comb = np.concatenate([X1, X0])
    # print("X_comb", X_comb.shape)
    X_des = np.concatenate([np.ones((n,1)), X_comb], axis=1)
    # print("X_des", X_des.shape)
    T = np.concatenate([np.ones(n1, dtype=np.int32), np.zeros(n0, dtype=np.int32)])
    Y = np.zeros_like(T)

    if outcome_type == "linear":
        # _Y0 = beta0[0] - X_comb @ beta0[1:]
        # print("X_comb @ beta0[1:]", (_Y0).shape)
        # _Y1 = beta1[0] + X_comb @ beta1[1:]
        # _Y0 = (beta0[0] - .5 * (X_comb @ beta0[1:])).flatten()
        # _Y0 = beta0[0] - 0.5 * X_comb @ beta0[1:]
        outcome_info = {
        'type': 'linear',
        'beta_0': beta0,
        'beta_1': beta1
        }

        _Y0 = X_des @ beta0
        _Y1 = X_des @ beta1 
        # print(X_comb.mean(0) @ beta1[1:])
        # true ATE
        # tau = beta1[0] - beta0[0] # + X_comb.mean(0) @ beta1[1:] + 0.5 * X_comb.mean(0) @ beta0[1:]
    elif outcome_type == "quadratic":
        beta0[1:] = -.5 * beta0[1:]
        _Y0 = X_des @ beta0 + X_comb**2 @ beta0[1:]
        _Y1 = X_des @ beta1 + X_comb**2 @ beta1[1:]
        outcome_info = {
        'type': 'quadratice',
        'beta_0': beta0,
        'beta_1': beta1
        }
        # _Y0 = (beta0[0] - (X_comb @ beta0[1:]) + X_comb**2 @ beta0[1:]).flatten()
        # _Y1 = X_des @ beta1 + X_comb**2 @ beta1[1:]
        # true ATE

    Y[T==1] = _Y1[T==1] + np.random.normal(size=(n1,)) * sigma
    Y[T==0] = _Y0[T==0] + np.random.normal(size=(n0,)) * sigma

    # true ATE
    true_ATT = (_Y1[T==1] - _Y0[T==1]).mean()

    return {
    'X_control': X0,
    'X_treatment': X1,
    'Y_control': Y[T==0],
    'Y_treatment': Y[T==1],
    'Y0_treatment': _Y0[T==1],
    'true_ATT': true_ATT,
    'outcome_info': outcome_info,
    'n0': n0,
    'n1': n1,
    'd': d
    }   

    # # # randomly split the sample into two parts
    # # idx = np.random.choice(n, n//2, replace=False)
    # # mask = ~np.isin(np.arange(n), idx)

    # # randomly split the sample into two parts - stratified
    # treated_idx = np.where(T == 1)[0]
    # control_idx = np.where(T == 0)[0]
    
    # n_treated_1 = len(treated_idx) // 2
    # n_control_1 = len(control_idx) // 2

    # treated_1_idx = np.random.choice(treated_idx, n_treated_1, replace=False)
    # control_1_idx = np.random.choice(control_idx, n_control_1, replace=False)

    # idx = np.concatenate([treated_1_idx, control_1_idx])
    # # Create mask for second part
    # mask = ~np.isin(np.arange(n), idx)

    # return X_comb[idx], X_comb[mask], Y[idx], Y[mask], T[idx], T[mask], tau
    # return X_comb, T, Y, tau


def gen_norm_data(rng=np.random.default_rng(0), n0=300, n1=300, d=2,
                 overlap_param=3.0, outcome_type='linear', 
                 sigma=0.01, plot=False):
    n = n1 + n0
    # def _model(n, r, beta0, beta1, s=0, sigma0=.01, linear=True, plot=False):
    beta1 = np.array([-1.1] + [2.5]*d)
    beta0 = np.array([1.2] + [2.5]*d)

    a1, b1 = 1.3, .3
    a0, b0 = .3, 2.

    X1 = rng.normal(size=(n1,d)) * 5 - overlap_param + rng.normal(size=(n1,d))
    X0 = rng.normal(size=(n0,d)) * 5 + overlap_param + rng.normal(size=(n0,d))

    # X1 = np.random.normal(size=(n1,d)) * (2*s+1) + np.random.normal(size=(n1,d))
    # X0 = np.random.normal(size=(n0,d)) * 1 + np.random.normal(size=(n0,d))

    if plot:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(X1[:,0], label=r'$P_X^1$', fill=True, alpha=0.5)
        sns.kdeplot(X0[:,0], label=r'$P_X^0$', fill=True, alpha=0.5)

        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        plt.close()

    X_comb = np.concatenate([X1, X0])
    # print("X_comb", X_comb.shape)
    X_des = np.concatenate([np.ones((n,1)), X_comb], axis=1)
    T = np.concatenate([np.ones(n1, dtype=np.int32), np.zeros(n0, dtype=np.int32)])
    Y = np.zeros_like(T)

    if outcome_type == "linear":
        _Y0 = X_des @ beta0
        _Y1 = X_des @ beta1 
        outcome_info = {
        'type': 'linear',
        'beta_0': beta0,
        'beta_1': beta1
        }
    elif outcome_type == "quadratic":
        beta0[1:] = -.5 * beta0[1:]
        _Y0 = X_des @ beta0 + X_comb**2 @ beta0[1:]
        _Y1 = X_des @ beta1 + X_comb**2 @ beta1[1:]
        outcome_info = {
        'type': 'quadratice',
        'beta_0': beta0,
        'beta_1': beta1
        }
    # _Y0 = (beta0[0] - 0.5 * (X_comb @ beta0[1:])).flatten()
    # _Y1 = X_des @ beta1 
    Y[T==1] = _Y1[T==1] + np.random.normal(size=(n1,)) * sigma
    Y[T==0] = _Y0[T==0] + np.random.normal(size=(n0,)) * sigma 

    # true ATE
    true_ATT = (_Y1[T==1] - _Y0[T==1]).mean()

    return {
    'X_control': X0,
    'X_treatment': X1,
    'Y_control': Y[T==0],
    'Y_treatment': Y[T==1],
    'Y0_treatment': _Y0[T==1],
    'true_ATT': true_ATT,
    'outcome_info': outcome_info,
    'n0': n0,
    'n1': n1,
    'd': d
    }   

    # # randomly split the sample into two parts
    # idx = np.random.choice(n, n//2, replace=False)
    # mask = ~np.isin(np.arange(n), idx)

    # return X_comb[idx], X_comb[mask], T[idx], T[mask], Y[idx], Y[mask], tau




def standardize_data(data_dict: Dict, 
                     method: str = 'zscore') -> Tuple[Dict, Dict]:
    """
    Standardize covariates using COMBINED statistics.
    
    CRITICAL: Always use combined (treatment + control) statistics!
    """
    
    X0 = data_dict['X_control'].copy()
    X1 = data_dict['X_treatment'].copy()
    
    # Use combined data for scaling parameters
    X_combined = np.vstack([X0, X1])
    
    if method == 'zscore':
        mean = X_combined.mean(axis=0)
        std = X_combined.std(axis=0, ddof=1)
        std[std == 0] = 1.0  # Avoid division by zero
        
        X0_std = (X0 - mean) / std
        X1_std = (X1 - mean) / std
        
        scaler_params = {'mean': mean, 'std': std, 'method': 'zscore'}
        
    elif method == 'minmax':
        min_val = X_combined.min(axis=0)
        max_val = X_combined.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        X0_std = (X0 - min_val) / range_val
        X1_std = (X1 - min_val) / range_val
        
        scaler_params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
    elif method == 'robust':
        median = np.median(X_combined, axis=0)
        q75 = np.percentile(X_combined, 75, axis=0)
        q25 = np.percentile(X_combined, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        
        X0_std = (X0 - median) / iqr
        X1_std = (X1 - median) / iqr
        
        scaler_params = {'median': median, 'iqr': iqr, 'method': 'robust'}
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create standardized data dict
    standardized_dict = data_dict.copy()
    standardized_dict['X_control'] = X0_std
    standardized_dict['X_treatment'] = X1_std
    
    return standardized_dict, scaler_params


def visualize_overlap(data, estimand='ATT'):
    """
    Visualize propensity score overlap for LaLonde data.
    Requires sklearn for logistic regression.
    """
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    
    # Combine data
    X = np.vstack([data['X_control'], data['X_treatment']])
    T = np.concatenate([np.zeros(data['n0']), np.ones(data['n1'])])
    
    # Fit propensity score model
    ps_model = LogisticRegression(max_iter=1e7)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    
    # Split by treatment
    ps_control = ps[T == 0]
    ps_treated = ps[T == 1]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(ps_control, bins=30, alpha=0.6, label='Control', density=True, color='blue')
    plt.hist(ps_treated, bins=30, alpha=0.6, label='Treated', density=True, color='red')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title(f'Propensity Score Overlap ({data["outcome_info"]["version"]})')
    plt.legend()
    plt.axvline(x=0.1, color='gray', linestyle='--', label='Common trim: 0.1')
    plt.axvline(x=0.9, color='gray', linestyle='--', label='Common trim: 0.9')
    plt.grid(alpha=0.3)
    
    # Calculate overlap metrics
    min_treated = ps_treated.min()
    max_treated = ps_treated.max()
    min_control = ps_control.min()
    max_control = ps_control.max()
    
    print(f"\nPropensity Score Range:")
    print(f"  Treated: [{min_treated:.3f}, {max_treated:.3f}]")
    print(f"  Control: [{min_control:.3f}, {max_control:.3f}]")
    print(f"  Common support: [{max(min_treated, min_control):.3f}, "
          f"{min(max_treated, max_control):.3f}]")
    
    # Count violations
    if estimand == 'ATT':
        violations = np.sum(ps_control > 0.95)
        print(f"\nATT Violations (P(T=1|X) > 0.95 in controls): {violations}/{len(ps_control)}")
    elif estimand == 'ATE':
        violations_low = np.sum(ps < 0.05)
        violations_high = np.sum(ps > 0.95)
        print(f"\nATE Violations:")
        print(f"  P(T=1|X) < 0.05: {violations_low}/{len(ps)}")
        print(f"  P(T=1|X) > 0.95: {violations_high}/{len(ps)}")
    
    plt.show()
    return ps


# Example usage:
if __name__ == "__main__":
    # Load semi-synthetic data with poor overlap
    # Step 1: Generate semi-synthetic data (do this once)
    delta = 2.
    semisynthetic_path = f"data/rhc_semisynthetic_{delta}.csv"
    metadata = create_semisynthetic_rhc(
        original_path="data/rhc_processed.csv",
        save_path=semisynthetic_path,
        true_att=-0.1,
        overlap_strength=delta,
        propensity_type='linear',
        outcome_type='heterogeneous',
        seed=42,
        include_propensity_scores=True,
        include_counterfactuals=True
    )

    # # Step 2: Load and use semi-synthetic data
    # data = load_rhc_data(
    #     path="data/rhc_processed.csv",
    #     standardize=True,
    #     semisynthetic=True,
    #     semisynthetic_path=semisynthetic_path
    # )
    # # print(data.head())
    # print(f"True ATT: {data['true_ATT']:.4f}")

    # print(data['n1'])
    # print(data['n0'])

    # Step 4: Generate multiple versions with different overlap
    for overlap in np.linspace(0,1,5):
        semisynthetic_path = f"data/rhc_semisynthetic_{(overlap)}.csv"
        create_semisynthetic_rhc(
            original_path="data/rhc_processed.csv",
            save_path=semisynthetic_path,
            true_att=-1.,
            overlap_strength=overlap,
            propensity_type='nonlinear',
            outcome_type='heterogeneous',
            seed=42,
            include_propensity_scores=True,
            include_counterfactuals=True
        )

    # Visualize overlap
    # fig = visualize_overlap_rhc(metadata)
    # plt.show()

    # Now test your method and compare to true ATT!

    # load_rhc_data(path="data/rhc_processed.csv", download=False)

    # # Load experimental data (gold standard)
    # print("="*60)
    # print("EXPERIMENTAL DATA (RCT - Ground Truth)")
    # print("="*60)
    # data_exp = load_lalonde_data(
    #     path="data/lalond_experimental.csv", download=True, 
    #     version='experimental', include_categorical=False)
    
    # print("\n" + "="*60)
    # print("OBSERVATIONAL DATA - PSID (Severe Overlap Issues)")
    # print("="*60)
    # data_psid = load_lalonde_data(
    #     path="data/lalond_psid.csv", download=False,
    #     version='psid', include_categorical=False)
 
    # Visualize overlap (requires sklearn and matplotlib)
    # visualize_overlap(data_psid, estimand='ATT')