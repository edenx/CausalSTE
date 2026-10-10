import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
# warnings.filterwarnings('ignore')
from .crossfit import *
from ..data_utils.gen_data import *
import matplotlib.pyplot as plt
import seaborn as sns


def create_features(X, order=1):
    """Create polynomial features up to specified order."""
    if order == 1:
        return X
    elif order == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        return poly.fit_transform(X)
    else:
        raise ValueError("Order must be 1 or 2")


def estimate_ATT_crossfit(crossfit_data, method='ipw', ps_order=1, verbose=False):
    """
    Estimate ATT using cross-fitting with pre-computed folds.
    
    Parameters:
    -----------
    crossfit_data : CrossFitData
        Pre-computed data splits
    method : str
        Method to use: 'ipw', 'sipw', 'aipw', 'or', 'did'
    ps_order : int
        Order of logistic regression for propensity score (1 or 2)
    or_order : int
        Order of regression for outcome model (1 or 2)
    verbose : bool
        Print detailed information
    
    Returns:
    --------
    results : dict
        ATT estimates, bias, etas, and detailed fold results
    """
    K = crossfit_data.K
    n1, d = crossfit_data.X_treatment.shape
    n0 = len(crossfit_data.X_control)
    true_ATT = crossfit_data.true_ATT
    
    # Storage for results
    fold_results = []
    all_Y0_hat = np.zeros(n1)
    all_X_mapped = np.zeros((n1, d))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Cross-Fitting ATT Estimation")
        print(f"  Method: {method.upper()}")
        if method in ['ipw', 'sipw', 'aipw']:
            print(f"  Propensity Score Order: {ps_order}")
        print(f"  K = {K} folds")
        print(f"  n_treatment = {n1}, n_control = {n0}")
        print(f"{'='*60}")
    
    # Iterate over folds
    for fold_idx in range(K):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{K} ---")
        
        # Get fold data
        fold_data = crossfit_data.get_fold_data(fold_idx)
        
        X_ctrl_train = fold_data['X_ctrl_train']
        Y_ctrl_train = fold_data['Y_ctrl_train']
        X_trt_train = fold_data['X_trt_train']
        Y_trt_train = fold_data['Y_trt_train']
        X_trt_test = fold_data['X_trt_test']
        Y_trt_test = fold_data['Y_trt_test']
        Y0_trt_test = fold_data['Y0_trt_test']
        test_trt_idx = fold_data['test_trt_idx']
        
        n_test = len(X_trt_test)
        
        if verbose:
            print(f"  Training: {len(X_ctrl_train)} control, {len(X_trt_train)} treatment")
            print(f"  Testing:  {n_test} treatment")
        
        # Apply method-specific estimation
        if method == 'ipw':
            ATT_fold, Y0_hat_fold, var_fold = estimate_fold_ipw(
                X_ctrl_train, Y_ctrl_train, X_trt_train, Y_trt_train,
                X_trt_test, Y_trt_test, ps_order)
            
        elif method == 'sipw':
            ATT_fold, Y0_hat_fold, var_fold = estimate_fold_sipw(
                X_ctrl_train, Y_ctrl_train, X_trt_train, Y_trt_train,
                X_trt_test, Y_trt_test, ps_order, n0, n1)
            
        elif method == 'or':
            outcome_model = fold_data['outcome_model']
            ATT_fold, Y0_hat_fold, var_fold = estimate_fold_or(
                X_trt_test, Y_trt_test, outcome_model)
            
        elif method == 'aipw':
            outcome_model = fold_data['outcome_model']
            ATT_fold, Y0_hat_fold, var_fold = estimate_fold_aipw(
                X_ctrl_train, Y_ctrl_train, X_trt_train, Y_trt_train,
                X_trt_test, Y_trt_test, outcome_model, ps_order)
            
        elif method == 'did':
            # DiD: simple difference between test treatment and train control
            ATT_fold, Y0_hat_fold, var_fold = estimate_fold_did(
                Y_ctrl_train, Y_trt_test)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if true_ATT is not None:
            # Compute fold-specific metrics
            bias_fold = ATT_fold - true_ATT
        else:
            bias_fold = None
        
        # Store fold results
        fold_results.append({
            'ATT_estimate': ATT_fold,
            'bias': bias_fold,
            'variance': var_fold,
            'n_samples': n_test,
        })
        
        # Store individual predictions
        if Y0_hat_fold is not None:
            all_Y0_hat[test_trt_idx] = Y0_hat_fold
        all_X_mapped[test_trt_idx] = X_trt_test  # Store original X for consistency
        
        if verbose:
            print(f"  ATT estimate: {ATT_fold:.4f}")
            print(f"  Bias: {bias_fold}")
            print(f"  Variance: {var_fold:.6f}")
    
    # Aggregate results across folds
    # Cross-fitted ATT estimate (simple average across folds)
    ATT_estimate = np.mean([fold['ATT_estimate'] for fold in fold_results])
    
    # Weighted average by sample size
    total_samples = sum(fold['n_samples'] for fold in fold_results)
    ATT_estimate_weighted = sum(fold['ATT_estimate'] * fold['n_samples'] 
                               for fold in fold_results) / total_samples
    
    
    fold_variances = np.array([fold['variance'] for fold in fold_results])
    variance = np.sum(fold_variances) / (K**2)

    if true_ATT is not None:
        bias = ATT_estimate - true_ATT
        # MSE = Bias^2 + Variance
        mse = bias**2 + variance
    else:
        bias = None
        mse = None

    # Overall prediction metrics (for OR and AIPW)
    if method in ['or', 'aipw'] and crossfit_data.Y0_treatment is not None:
        prediction_error = np.mean((all_Y0_hat - crossfit_data.Y0_treatment) ** 2)
    else:
        prediction_error = np.nan
    
    results = {
        'ATT_estimate': ATT_estimate,
        'ATT_estimate_weighted': ATT_estimate_weighted,
        'true_ATT': true_ATT,
        'bias': bias,
        'variance': variance,
        'mse': mse,
        'X_mapped': all_X_mapped,
        'Y0_hat': all_Y0_hat,
        'Y0_true': crossfit_data.Y0_treatment,
        'prediction_MSE': prediction_error,
        'K': K,
        'fold_results': fold_results,
        'fold_std': np.std([fold['ATT_estimate'] for fold in fold_results]),
        'method': method,
        'ps_order': ps_order if method in ['ipw', 'sipw', 'aipw'] else None,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Cross-Fitted Results (K={K}):")
        print(f"  True ATT: {true_ATT}")
        print(f"  Estimated ATT: {ATT_estimate:.4f}")
        print(f"  Bias: {bias}")
        print(f"  Variance: {variance:.6f}")
        print(f"  MSE (Bias² + Var): {mse}")
        print(f"  Std across folds: {results['fold_std']:.4f}")
        if not np.isnan(prediction_error):
            print(f"  Prediction MSE: {prediction_error:.4f}")
        print(f"{'='*60}")
    
    return results


def estimate_fold_ipw(X_ctrl_train, Y_ctrl_train, X_trt_train, Y_trt_train,
                      X_trt_test, Y_trt_test, ps_order):
    """
    Estimate ATT for a single fold using IPW.
    
    Train: Fit propensity model on train_ctrl + train_trt
    Test: Evaluate on test_trt, using weighted train_ctrl for counterfactual
    
    Returns: ATT_fold, etas_fold, Y0_hat_fold, variance_fold
    """
    # Combine train data to fit propensity model
    X_combined = np.vstack([X_trt_train, X_ctrl_train])
    T_combined = np.concatenate([np.ones(len(X_trt_train)), np.zeros(len(X_ctrl_train))])
    
    # Create features and fit propensity model
    X_features = create_features(X_combined, order=ps_order)
    ps_model = LogisticRegression(penalty=None, max_iter=int(1e7), solver='lbfgs')
    ps_model.fit(X_features, T_combined)
    
    # Get propensity scores for train control (used for weighting)
    X_ctrl_features = create_features(X_ctrl_train, order=ps_order)
    ps_control = ps_model.predict_proba(X_ctrl_features)[:, 1]
    # ps_control = np.clip(ps_control, 0.01, 0.99)
    
    # # Get propensity scores for test treatment (returned as etas)
    # X_trt_test_features = create_features(X_trt_test, order=ps_order)
    # ps_test = ps_model.predict_proba(X_trt_test_features)[:, 1]
    # ps_test = np.clip(ps_test, 0.01, 0.99)
    
    # IPW weights for control group
    weights = ps_control / (1 - ps_control)
    
    # ATT estimate: E[Y|T=1] on test - weighted E[Y|T=0] on train
    term1 = np.mean(Y_trt_test)
    term2 = np.sum(weights * Y_ctrl_train) / np.sum(weights)
    ATT_fold = term1 - term2
    
    # Variance estimation using influence function approach
    n_test = len(Y_trt_test)
    
    # Variance of term1 (treated outcomes)
    var_term1 = np.var(Y_trt_test, ddof=1) / n_test
    
    # Variance of term2 (weighted control outcomes)
    # Using Horvitz-Thompson variance estimator
    weighted_residuals = weights * (Y_ctrl_train - term2)
    var_term2 = np.sum(weighted_residuals**2) / (np.sum(weights)**2)
    
    # Total variance
    variance_fold = var_term1 + var_term2
    
    # Y0_hat: IPW estimate of E[Y(0)|T=1] for all test samples
    Y0_hat = np.full(len(X_trt_test), term2)
    
    return ATT_fold, Y0_hat, variance_fold


def estimate_fold_sipw(X_ctrl_train, Y_ctrl_train, X_trt_train, Y_trt_train,
                       X_trt_test, Y_trt_test, ps_order, n0, n1):
    """
    Estimate ATT for a single fold using Stabilized IPW.
    
    Train: Fit propensity model on train_ctrl + train_trt
    Test: Evaluate on test_trt, using weighted train_ctrl for counterfactual
    
    Returns: ATT_fold, etas_fold, Y0_hat_fold, variance_fold
    """
    # Combine train data to fit propensity model
    X_combined = np.vstack([X_trt_train, X_ctrl_train])
    T_combined = np.concatenate([np.ones(len(X_trt_train)), np.zeros(len(X_ctrl_train))])
    
    # Create features and fit propensity model
    X_features = create_features(X_combined, order=ps_order)
    ps_model = LogisticRegression(penalty=None, max_iter=int(1e7), solver='lbfgs')
    ps_model.fit(X_features, T_combined)
    
    # Get propensity scores for train control
    X_ctrl_features = create_features(X_ctrl_train, order=ps_order)
    ps_control = ps_model.predict_proba(X_ctrl_features)[:, 1]
    # ps_control = np.clip(ps_control, 0.01, 0.99)
    
    # # Get propensity scores for test treatment (returned as etas)
    # X_trt_test_features = create_features(X_trt_test, order=ps_order)
    # ps_test = ps_model.predict_proba(X_trt_test_features)[:, 1]
    # ps_test = np.clip(ps_test, 0.01, 0.99)
    
    # SIPW weights (stabilized)
    p_treat = n1 / (n1 + n0)
    weights = (p_treat * ps_control) / ((1 - p_treat) * (1 - ps_control))
    
    # Normalize weights
    weights = weights / np.sum(weights) * len(weights)
    
    # ATT estimate
    term1 = np.mean(Y_trt_test)
    term2 = np.sum(weights * Y_ctrl_train) / len(weights)
    ATT_fold = term1 - term2
    
    # Variance estimation
    n_test = len(Y_trt_test)
    
    # Variance of term1
    var_term1 = np.var(Y_trt_test, ddof=1) / n_test
    
    # Variance of term2 (stabilized weights)
    weighted_residuals = weights * (Y_ctrl_train - term2)
    var_term2 = np.sum(weighted_residuals**2) / (len(weights)**2)
    
    # Total variance
    variance_fold = var_term1 + var_term2
    
    # Y0_hat: SIPW estimate of E[Y(0)|T=1] for all test samples
    Y0_hat = np.full(len(X_trt_test), term2)
    
    return ATT_fold, Y0_hat, variance_fold


def estimate_fold_or(X_trt_test, Y_trt_test, outcome_model):
    """
    Estimate ATT for a single fold using Outcome Regression.
    
    Train: Outcome model already fitted on train_ctrl (by CrossFitData)
    Test: Predict Y(0) for test_trt using RAW features
    
    Note: outcome_model may be a Pipeline that handles feature transformation,
    so we pass RAW features.
    
    Returns: ATT_fold, etas_fold, Y0_hat_fold, variance_fold
    """
    # Predict Y(0) for test treatment group using RAW features
    # If outcome_model is a Pipeline, it will transform internally
    Y0_hat = outcome_model.predict(X_trt_test)
    
    # ATT estimate: E[Y(1) - Y(0) | T=1] on test
    treatment_effects = Y_trt_test - Y0_hat
    ATT_fold = np.mean(treatment_effects)
    
    # Variance estimation: Var(Y - μ(X)) / n
    n_test = len(Y_trt_test)
    variance_fold = np.var(treatment_effects, ddof=1) / n_test
    
    
    return ATT_fold, Y0_hat, variance_fold


def estimate_fold_aipw(X_ctrl_train, Y_ctrl_train, X_trt_train, Y_trt_train,
                       X_trt_test, Y_trt_test, outcome_model, ps_order):
    """
    Estimate ATT for a single fold using AIPW (Doubly Robust).
    
    Train: Fit propensity model on train_ctrl + train_trt
           Outcome model already fitted on train_ctrl (by CrossFitData)
    Test: Evaluate on test_trt with doubly robust correction
    
    Note: outcome_model may be a Pipeline, so we pass RAW features.
    
    Returns: ATT_fold, etas_fold, Y0_hat_fold, variance_fold
    """
    # Combine train data to fit propensity model
    X_combined = np.vstack([X_trt_train, X_ctrl_train])
    T_combined = np.concatenate([np.ones(len(X_trt_train)), np.zeros(len(X_ctrl_train))])
    
    # Create features and fit propensity model
    X_features = create_features(X_combined, order=ps_order)
    ps_model = LogisticRegression(penalty=None, max_iter=int(1e7), solver='lbfgs')
    ps_model.fit(X_features, T_combined)
    
    # Get propensity scores for train control
    X_ctrl_features_ps = create_features(X_ctrl_train, order=ps_order)
    ps_control = ps_model.predict_proba(X_ctrl_features_ps)[:, 1]
    # ps_control = np.clip(ps_control, 0.01, 0.99)
    
    # # Get propensity scores for test treatment (returned as etas)
    # X_trt_test_features_ps = create_features(X_trt_test, order=ps_order)
    # ps_test = ps_model.predict_proba(X_trt_test_features_ps)[:, 1]
    # ps_test = np.clip(ps_test, 0.01, 0.99)
    
    # Get outcome predictions using RAW features
    Y0_hat_test = outcome_model.predict(X_trt_test)
    Y0_hat_control = outcome_model.predict(X_ctrl_train)
    
    # AIPW estimator for ATT
    # ATT = E[Y(1) | T=1] - E[Y(0) | T=1]
    # term1: E[Y(1) | T=1] on test
    term1 = np.mean(Y_trt_test)
    
    # term2: Doubly robust estimate of E[Y(0) | T=1]
    weights = ps_control / (1 - ps_control)
    term2_or = np.mean(Y0_hat_test)  # Outcome regression component on test
    term2_correction = np.sum(weights * (Y_ctrl_train - Y0_hat_control)) / np.sum(weights)
    
    term2 = term2_or + term2_correction
    
    ATT_fold = term1 - term2
    
    # Variance estimation for AIPW
    # Using influence function: IF = (Y - μ(X)) - w*(Y - μ(X)) from control
    n_test = len(Y_trt_test)
    
    # Variance from treated
    var_term1 = np.var(Y_trt_test, ddof=1) / n_test
    
    # Variance from OR component
    var_term2_or = np.var(Y0_hat_test, ddof=1) / n_test
    
    # Variance from correction term
    weighted_residuals = weights * (Y_ctrl_train - Y0_hat_control)
    var_term2_correction = np.sum(weighted_residuals**2) / (np.sum(weights)**2)
    
    # Total variance (approximate - ignores covariance terms)
    variance_fold = var_term1 + var_term2_or + var_term2_correction
    
    return ATT_fold, Y0_hat_test, variance_fold


def estimate_fold_did(Y_ctrl_train, Y_trt_test):
    """
    Estimate ATT for a single fold using Difference-in-Differences.
    
    Note: This is a simplified DiD without pre-treatment data.
    
    Returns: ATT_fold, etas_fold, Y0_hat_fold, variance_fold
    """
    # Simple difference
    term1 = np.mean(Y_trt_test)
    term2 = np.mean(Y_ctrl_train)
    ATT_fold = term1 - term2
    
    # Variance estimation
    n_test = len(Y_trt_test)
    n_ctrl = len(Y_ctrl_train)
    
    var_term1 = np.var(Y_trt_test, ddof=1) / n_test
    var_term2 = np.var(Y_ctrl_train, ddof=1) / n_ctrl
    
    variance_fold = var_term1 + var_term2
    
    # No propensity scores or individual predictions for DiD
    Y0_hat = np.full(n_test, term2)
    
    return ATT_fold, Y0_hat, variance_fold


# Example usage and testing
if __name__ == "__main__":
    
    
    # Generate data
    data = gen_beta_data(rng=np.random.default_rng(42), n0=300, n1=300, d=2, 
                         overlap_param=3.0, outcome_type='linear', sigma=0.01, plot=False)
    
    print(f"True ATT: {data['true_ATT']:.4f}\n")
    
    # Create CrossFitData object
    crossfit_data = CrossFitData(
        X_control=data['X_control'],
        X_treatment=data['X_treatment'],
        Y_control=data['Y_control'],
        Y_treatment=data['Y_treatment'],
        Y0_treatment=data['Y0_treatment'],
        true_ATT=data['true_ATT'],
        K=5,
        random_state=42
    )
    
    crossfit_data.fit_transport_maps(
            verbose=True, method='exact', oos_method='auto')
    crossfit_data.fit_outcome_models(outcome_model_type='linear')
    # Test all methods
    methods = ['ipw', 'sipw', 'or', 'aipw']
    
    print("="*70)
    print("CROSS-FITTING RESULTS (K=5)")
    print("="*70)
    
    for method in methods:
        if method in ['ipw', 'sipw']:
            # Test with 1st and 2nd order propensity score
            for ps_order in [1,2]:
                print(f"\n\n{'='*70}")
                print(f"Method: {method.upper()}, PS Order: {ps_order}")
                print('='*70)
                results = estimate_ATT_crossfit(crossfit_data, method=method, 
                                               ps_order=ps_order, verbose=True)
                
        elif method == 'or':
            # Test with 1st and 2nd order outcome regression
            for or_order in [1]:
                print(f"\n\n{'='*70}")
                print(f"Method: {method.upper()}, OR Order: {or_order}")
                print('='*70)
                results = estimate_ATT_crossfit(crossfit_data, method=method, 
                                               verbose=True)
                
        elif method == 'aipw':
            # Test all combinations
            for ps_order in [1,2]:
                for or_order in [1]:
                    print(f"\n\n{'='*70}")
                    print(f"Method: {method.upper()}, PS Order: {ps_order}, OR Order: {or_order}")
                    print('='*70)
                    results = estimate_ATT_crossfit(crossfit_data, method=method, 
                                                   ps_order=ps_order, 
                                                   verbose=True)
    
    # Test DiD
    print(f"\n\n{'='*70}")
    print("Method: DID")
    print('='*70)
    results_did = estimate_ATT_crossfit(crossfit_data, method='did', verbose=True)



