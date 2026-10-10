import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import ot
import pandas as pd
from .utils import extract_linear_params, extract_quadratic_params, gradient_eval
# from ot_maps import OptimalTransportEstimator
from .crossfit import CrossFitData, bootstrap
from .data_utils.gen_data import *
from .data_utils.gen_rhc import *
from .data_utils.gen_lalonde import *
from .estimator import MultidimRatioEstimator, MultidimIPMEstimator, _ipm
from .att_other import estimate_ATT_crossfit
import warnings


class ATTEstimatorWithOT:
    """
    Estimate ATT (Average Treatment Effect on the Treated) using outcome regression
    with optimal transport-based covariate mapping to handle overlap violations.
    """
    
    def __init__(
        self, 
        method='sinkhorn', 
        oos_method='mapping_gaussian', 
        reg=0.05, 
        outcome_model_type='linear', 
        mapping_params=None,
        seed=42):
        """
        Parameters:
        -----------
        method : str, 'exact' or 'sinkhorn'
        reg : float, regularization for Sinkhorn
        outcome_model_type : str, 'linear' or 'quadratic'
            Model used for outcome regression (may differ from DGP)
        """
        self.method = method
        self.oos_method = oos_method
        self.reg = reg
        self.mapping_params = mapping_params
        self.outcome_model_type = outcome_model_type
        self.inverse_transport_plan = None
        self.outcome_model = None
        # self.scaler = StandardScaler()
        self.rng = np.random.default_rng(seed)
        
    
    def estimate_inverse_transport(self, X_control, X_treatment, verbose=False):
        self.ot_estimator.estimate_inverse_transport(X_control, X_treatment, verbose=verbose)
        self.inverse_transport_plan = self.ot_estimator.ot_estimator
    
    def map_treatment_to_control(self, X_treatment, idx=None):
        return self.ot_estimator.map_treatment_to_control(X_treatment)
        # """Map treatment sample idx to control space"""
        # Pi = self.inverse_transport_plan
        # if idx is not None:
        #     weights = Pi[idx, :]
        #     weights = weights / weights.sum()
        #     return weights @ X_control
        # else:
        #     return Pi @ X_control / Pi.sum(1)[:,None]

    def find_optimal_eta(self, x1, x0, X_control, verbose=False, 
                               tol=1e-10, max_attempts=3):
        """
        Find optimal η using LP to ensure x_mapped in convex hull.
        
        Improvements:
        - Better numerical stability for large sample sizes
        - Adaptive tolerance
        - Multiple solver attempts with different parameters
        - Detailed diagnostics
        - Early detection of infeasible problems
        """
        n0, d = X_control.shape
        
        if verbose:
            print(f"\n--- Finding optimal η (n0={n0}, d={d}) ---")
        
        # ============================================================
        # STEP 1: Check if x1 is already in convex hull
        # ============================================================
        
        # Problem: Check if x1 = Σ λᵢ xᵢ^control with Σλᵢ=1, λᵢ≥0
        c_test = np.zeros(n0)
        
        # Build constraint matrix: X_control^T @ λ = x1
        A_eq_test = np.vstack([X_control.T, np.ones(n0)])
        b_eq_test = np.concatenate([x1, [1.0]])
        
        bounds_test = [(0, None)] * n0
        
        # Try with adaptive tolerance
        for attempt in range(max_attempts):
            current_tol = tol * (10 ** attempt)
            
            result_test = linprog(
                c_test, 
                A_eq=A_eq_test, 
                b_eq=b_eq_test, 
                bounds=bounds_test, 
                method='highs',
                options={'presolve': True, 'disp': False}
            )
            
            if result_test.success:
                # Check if solution is valid (all λᵢ ≥ -tol)
                if np.all(result_test.x >= -current_tol):
                    if verbose:
                        print(f"  x1 already in convex hull (attempt {attempt+1})")
                        print(f"  min(λ) = {np.min(result_test.x):.2e}")
                    return 0.0, x1.copy()
                elif verbose:
                    print(f"  Attempt {attempt+1}: Some λᵢ < 0, continuing...")
            elif verbose:
                print(f"  Attempt {attempt+1}: LP failed with status {result_test.status}")
        
        if verbose:
            print(f"  x1 NOT in convex hull, solving for minimal η...")
        
        # ============================================================
        # STEP 2: Solve for minimal η
        # ============================================================
        
        # Problem formulation:
        # min η
        # s.t. η·x0 + (1-η)·x1 = X_control @ λ
        #      Σλᵢ = 1
        #      λᵢ ≥ 0, 0 ≤ η ≤ 1
        #
        # Rewrite as:
        # η(x0 - x1) + x1 = X_control @ λ
        # => X_control @ λ - η(x0 - x1) = x1
        
        # Variables: [η, λ₁, ..., λₙ₀]
        n_vars = 1 + n0
        c = np.zeros(n_vars)
        c[0] = 1.0  # Minimize η
        
        # Build constraint matrix more efficiently
        diff = x0 - x1
        
        # Equality constraints: [-(x0-x1), X_control^T] @ [η, λ] = x1
        A_eq_top = np.column_stack([-diff.reshape(-1, 1), X_control.T])
        A_eq_bottom = np.concatenate([[0.0], np.ones(n0)]).reshape(1, -1)
        A_eq = np.vstack([A_eq_top, A_eq_bottom])
        b_eq = np.concatenate([x1, [1.0]])
        
        # Bounds: 0 ≤ η ≤ 1, λᵢ ≥ 0
        bounds = [(0, 1)] + [(0, None)] * n0
        
        # Check for numerical issues
        if verbose:
            cond_number = np.linalg.cond(A_eq)
            print(f"  Constraint matrix condition number: {cond_number:.2e}")
            if cond_number > 1e10:
                print(f" WARNING: Poor conditioning detected!")
        
        # Try multiple solver configurations
        solver_configs = [
            {'method': 'highs', 'options': {'presolve': True, 'disp': False}},
            {'method': 'highs-ds', 'options': {'presolve': True, 'disp': False}},
            {'method': 'highs-ipm', 'options': {'presolve': True, 'disp': False}},
        ]
        
        best_result = None
        best_eta = None
        
        for i, config in enumerate(solver_configs):
            try:
                result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, **config)
                
                if result.success:
                    eta = result.x[0]
                    lambdas = result.x[1:]
                    
                    # Validate solution
                    x_mapped = eta * x0 + (1 - eta) * x1
                    reconstruction = X_control.T @ lambdas
                    reconstruction_error = np.linalg.norm(x_mapped - reconstruction)
                    constraint_violation = np.abs(np.sum(lambdas) - 1.0)
                    
                    if verbose:
                        print(f"  Solver {config['method']} succeeded:")
                        print(f"    η = {eta:.6f}")
                        print(f"    Reconstruction error: {reconstruction_error:.2e}")
                        print(f"    Constraint violation: {constraint_violation:.2e}")
                        print(f"    min(λ) = {np.min(lambdas):.2e}")
                    
                    # Accept if errors are small
                    if reconstruction_error < 1e-4 and constraint_violation < 1e-6:
                        if best_result is None or eta < best_eta:
                            best_result = result
                            best_eta = eta
                            if i == 0:  # First solver worked, use it
                                break
                else:
                    if verbose:
                        print(f"  Solver {config['method']} failed: {result.message}")
                        
            except Exception as e:
                if verbose:
                    print(f"  Solver {config['method']} error: {str(e)}")
                continue
        
        # ============================================================
        # STEP 3: Return result or fallback
        # ============================================================
        
        if best_result is not None:
            eta = best_result.x[0]
            x_mapped = eta * x0 + (1 - eta) * x1
            
            if verbose:
                print(f"  ✓ Solution found: η = {eta:.6f}")
            
            return eta, x_mapped
        else:
            # All solvers failed - use fallback
            if verbose:
                print(f"  ✗ All solvers failed! Using fallback η=1.0")
            
            warnings.warn(
                f"LP solver failed for sample (n0={n0}). Using fallback η=1.0. "
                "This may indicate numerical issues or infeasible constraints.",
                RuntimeWarning
            )
            
            return 1.0, x0

    def diagnose_eta_problems(self, X_control, X_treatment, n_samples=10):
        """
        Run diagnostic tests to identify why find_optimal_eta might be failing.
        
        Returns dict with diagnostic information.
        """
        import time
        
        n0, d = X_control.shape
        n1 = X_treatment.shape[0]
        
        print("="*70)
        print("DIAGNOSTIC REPORT: find_optimal_eta")
        print("="*70)
        print(f"\nDataset info:")
        print(f"  n_control = {n0}")
        print(f"  n_treatment = {n1}")
        print(f"  dimensions = {d}")
        
        # Check data characteristics
        print(f"\nData characteristics:")
        print(f"  X_control range: [{X_control.min():.3f}, {X_control.max():.3f}]")
        print(f"  X_treatment range: [{X_treatment.min():.3f}, {X_treatment.max():.3f}]")
        print(f"  X_control std: {X_control.std():.3f}")
        print(f"  X_treatment std: {X_treatment.std():.3f}")
        
        # Check for duplicates or near-duplicates
        from scipy.spatial.distance import pdist
        if n0 <= 5000:  # Only for moderate sizes
            min_dist = pdist(X_control).min()
            print(f"  Min pairwise distance in X_control: {min_dist:.2e}")
            if min_dist < 1e-6:
                print(f"    ⚠️  WARNING: Near-duplicate points detected!")
        
        # Test on sample of treatment points
        n_test = min(n_samples, n1)
        test_indices = np.random.choice(n1, n_test, replace=False)
        
        # Estimate transport plan if needed
        if self.inverse_transport_plan is None:
            print(f"\nEstimating transport plan...")
            self.estimate_inverse_transport(X_control, X_treatment, verbose=True)
        
        print(f"\nTesting {n_test} random treatment samples:")
        print("-" * 70)
        
        successes = 0
        failures = 0
        times = []
        etas = []
        
        for i, idx in enumerate(test_indices):
            x1 = X_treatment[idx]
            x0 = self.map_treatment_to_control(X_control, X_treatment, idx)
            
            start = time.time()
            try:
                eta, x_mapped = self.find_optimal_eta(x1, x0, X_control)
                elapsed = time.time() - start
                
                times.append(elapsed)
                etas.append(eta)
                
                if eta >= 0.999:  # Likely a fallback
                    failures += 1
                    status = "⚠️  FALLBACK"
                else:
                    successes += 1
                    status = "✓"
                
                print(f"  Sample {i+1:2d}: η={eta:.4f}  time={elapsed:.3f}s  {status}")
                
            except Exception as e:
                failures += 1
                print(f"  Sample {i+1:2d}: ✗ EXCEPTION: {str(e)}")
        
        print("-" * 70)
        print(f"\nResults:")
        print(f"  Successes: {successes}/{n_test}")
        print(f"  Failures: {failures}/{n_test}")
        
        if times:
            print(f"  Mean time: {np.mean(times):.3f}s")
            print(f"  Max time: {np.max(times):.3f}s")
        
        if etas:
            print(f"  Mean η: {np.mean(etas):.4f}")
            print(f"  Median η: {np.median(etas):.4f}")
            print(f"  Max η: {np.max(etas):.4f}")
        
        print("="*70)
        
        return {
            'n_control': n0,
            'n_treatment': n1,
            'dimensions': d,
            'successes': successes,
            'failures': failures,
            'times': times,
            'etas': etas
        }
    # def find_optimal_eta(self, x1, x0, X_control):
    #     """Find optimal η using LP to ensure x_mapped in convex hull"""
    #     n0, d = X_control.shape
        
    #     # Check if x1 already in convex hull
    #     c_test = np.zeros(n0)
    #     A_eq_test = []
    #     b_eq_test = []
        
    #     for j in range(d):
    #         A_eq_test.append(X_control[:, j])
    #         b_eq_test.append(x1[j])
    #     A_eq_test.append(np.ones(n0))
    #     b_eq_test.append(1.0)
        
    #     A_eq_test = np.array(A_eq_test)
    #     b_eq_test = np.array(b_eq_test)
    #     bounds_test = [(0, None)] * n0
        
    #     result_test = linprog(c_test, A_eq=A_eq_test, b_eq=b_eq_test, 
    #                          bounds=bounds_test, method='highs')
        
    #     if result_test.success and np.all(result_test.x >= -1e-6):
    #         return 0.0, x1.copy()
        
    #     # Solve for minimal η
    #     c = np.zeros(1 + n0)
    #     c[0] = 1
        
    #     A_eq = []
    #     b_eq = []
        
    #     for j in range(d):
    #         row = np.zeros(1 + n0)
    #         row[0] = -(x0[j] - x1[j])
    #         row[1:] = X_control[:, j]
    #         A_eq.append(row)
    #         b_eq.append(x1[j])
        
    #     row = np.zeros(1 + n0)
    #     row[1:] = 1
    #     A_eq.append(row)
    #     b_eq.append(1.0)
        
    #     A_eq = np.array(A_eq)
    #     b_eq = np.array(b_eq)
    #     bounds = [(0, 1)] + [(0, None)] * n0
        
    #     result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
    #     if result.success:
    #         eta = result.x[0]
    #         x_mapped = eta * x0 + (1 - eta) * x1
    #         return eta, x_mapped
    #     else:
    #         return 1.0, x0
    

    def compute_all_individual_etas(self, X_control, X_treatment, verbose=False):
        """Compute individual optimal eta for each treatment sample"""
        n1 = X_treatment.shape[0]
        individual_etas = []

        if self.inverse_transport_plan is None:
            self.estimate_inverse_transport(X_control, X_treatment, verbose=verbose)
        
        X0 = self.map_treatment_to_control(X_treatment)
        for i in range(n1):
            x1 = X_treatment[i]
            x0 = X0[i]
            eta_i, _ = self.find_optimal_eta(x1, x0, X_control)
            individual_etas.append(eta_i)
        
        individual_etas = np.array(individual_etas)
        
        if verbose:
            print(f"\nIndividual η statistics:")
            print(f"  Min: {np.min(individual_etas):.3f}")
            print(f"  Max: {np.max(individual_etas):.3f}")
            print(f"  Mean: {np.mean(individual_etas):.3f}")
            print(f"  Median: {np.median(individual_etas):.3f}")
            print(f"  Std: {np.std(individual_etas):.3f}")
        
        return individual_etas


    # def estimate_ATT_with_eta(self, data, eta_strategy='optimal', 
    #                           global_eta=None, individual_etas=None, verbose=False,
    #                           metric='bias', eps = 1e-5):
    #     """
    #     Estimate ATT using outcome regression with specified η strategy.
        
    #     Parameters:
    #     -----------
    #     data : dict
    #         Output from generate_data
    #     eta_strategy : str
    #         'optimal' - use LP-based optimal η per sample
    #         'global' - use same η for all samples (requires global_eta parameter)
    #         None - no mapping (η=0 for all)
    #     global_eta : float
    #         If eta_strategy='global', use this η value
        
    #     Returns:
    #     --------
    #     results : dict with ATT estimate, bias, and individual etas
    #     """
    #     X_control = data['X_control']
    #     X_treatment = data['X_treatment']
    #     Y_control = data['Y_control']
    #     Y_treatment = data['Y_treatment']
    #     true_ATT = data['true_ATT']
        
    #     n1,d = X_treatment.shape
        
    #     # Estimate inverse transport
    #     if self.inverse_transport_plan is None:
    #         self.estimate_inverse_transport(X_control, X_treatment, verbose=verbose)
        
    #     # Fit outcome model on control group
    #     self.fit_outcome_model(X_control, Y_control)
    #     Y0_true = data['Y0_treatment']

    #     etas = []
        
    #     if eta_strategy == 'global':
    #         if global_eta is not None:
    #             eta = global_eta 
    #         else:
    #             eta, eta_grid, biases = self.find_best_global_eta(
    #                 data, eta_grid=None, metric=metric, verbose=False)
            
    #         X0 = self.map_treatment_to_control(X_control, X_treatment)
    #         X_mapped = eta * X0 + (1 - eta) * X_treatment
    #         etas.append(eta)

    #         # plt.figure()
    #         # plt.hist(X_treatment[:,0], label="treatment", alpha=0.5)
    #         # plt.hist(X_control[:,0], label="control", alpha=0.5)
    #         # plt.hist(X0[:,0], label="transported", alpha=0.5)
    #         # plt.hist(X_mapped[:,0], label="mapped", alpha=0.5)
    #         # plt.legend()
    #         # plt.title(f"strategy={eta_strategy}, eta={eta}")
    #         # plt.show()
    #         # plt.close()

    #     elif eta_strategy == 'optimal':
    #         X_mapped = []
    #         for i in range(n1):
    #             x1 = X_treatment[i]
    #             x0 = self.map_treatment_to_control(X_control, X_treatment, i)
                
    #             if individual_etas is not None:
    #                 eta = individual_etas[i]
    #                 x_mapped = eta * x0 + (1 - eta) * x1
    #             else:
    #                 eta, x_mapped = self.find_optimal_eta(x1, x0, X_control)

    #             X_mapped.append(x_mapped)
    #             etas.append(eta)
    #         X_mapped = np.array(X_mapped).reshape((n1, d))

    #     elif eta_strategy == None:
    #         etas.append(0)
    #         X_mapped = X_treatment

    #     elif eta_strategy == 'oracle':
    #         X0 = self.map_treatment_to_control(X_control, X_treatment)
    #         diff_X = (X0 - X_treatment)
    #         diff_Y = Y0_true - self.outcome_model.predict(X_treatment)
    #         if self.outcome_model_type == "linear":
    #             beta, c = extract_linear_params(self.outcome_model)
    #             gradf, _ = gradient_eval(func_type='linear', X=X_treatment, coef=beta, bias=c)
    #         print("diffy", (diff_Y).mean())
    #         print("diff_X", (diff_X @ gradf).mean())
    #         if np.abs(diff_Y.mean()) < eps or np.abs((diff_X @ gradf).mean()) < eps:
    #             eta = 0.
    #         else:
    #             eta = min(1, diff_Y.mean() / (diff_X @ gradf).mean())
    #             eta = max(eta, 0)
    #         print(f"estimated eta={eta}")
    #         X_mapped = eta * X0 + (1 - eta) * X_treatment
    #         etas.append(eta)

    #     else: 
    #         raise NotImplementedError("Use None, 'oracle', 'global', 'optimal'")
        
    #     etas = np.array(etas) 
        
    #     # Predict counterfactual outcomes for treatment group
    #     Y0_hat = self.outcome_model.predict(X_mapped)
        
    #     # ATT = E[Y(1) | A=1] - E[Y(0) | A=1]
    #     ATT_estimate = np.mean(Y_treatment) - np.mean(Y0_hat)
    #     bias = ATT_estimate - true_ATT
        
    #     # Compute true counterfactual prediction error
    #     prediction_error = np.mean((Y0_hat - Y0_true) ** 2)
    #     variance = np.var(Y0_hat)

    #     # print("variance", variance)
    #     # print("|bias|", np.abs(bias))
    #     # plt.figure()
    #     # plt.hist((Y0_hat - Y0_true)**2, bins=50)
    #     # plt.axvline(x=variance, label=f"var={variance:.3f}", color="red")
    #     # plt.axvline(x=bias**2, label=f"bias^2={bias**2: .3f}", color="green")
    #     # plt.legend()
    #     # plt.title(f"eta={eta}, eta_strategy={eta_strategy}")
    #     # plt.show()
    #     # plt.close()
        
    #     results = {
    #         'ATT_estimate': ATT_estimate,
    #         'true_ATT': true_ATT,
    #         'bias': bias,
    #         'variance': variance,
    #         'etas': etas,
    #         'mean_eta': np.mean(etas),
    #         'X_mapped': X_mapped,
    #         'Y0_hat': Y0_hat,
    #         'Y0_true': Y0_true,
    #         'prediction_MSE': prediction_error,
    #         'n_mapped': np.sum(etas > 0.01)
    #     }
    #     # print("true_ATT", true_ATT)
    #     # print("ATT Estimate", ATT_estimate)
    #     if verbose:
    #         print(f"\nATT Estimation ({eta_strategy}, model={self.outcome_model_type}):")
    #         print(f"  True ATT: {true_ATT:.4f}")
    #         print(f"  Estimated ATT: {ATT_estimate:.4f}")
    #         print(f"  Bias: {bias:.4f}")
    #         print(f"  Mean η: {np.mean(etas):.4f}")
    #         print(f"  Samples mapped: {np.sum(etas > 0.01)} / {n1}")
    #         # print(f"  Prediction MSE: {prediction_error:.4f}")
        
    #     return results


    def estimate_ATT_with_eta_crossfit(self, crossfit_data, eta_strategy='optimal', global_eta=None, 
                                       individual_etas=None, metric='bias', 
                                       eps=1e-2, verbose=False, max_avg_weight=10, gaussian=False,
                                       **kwargs):
        """
        Estimate ATT using cross-fitting with pre-computed folds and transport maps.
        
        Parameters:
        -----------
        crossfit_data : CrossFitData
            Pre-computed data splits and transport maps
        eta_strategy : str
            'optimal', 'global', None, or 'oracle'
        global_eta : float, optional
            Fixed eta value for 'global' strategy
        individual_etas : array, optional
            Pre-specified etas for each sample
        metric : str
            Metric for eta selection ('bias' or other)
        eps : float
            Small constant for numerical stability
        verbose : bool
            Print detailed information
        
        Returns:
        --------
        results : dict
            ATT estimates, bias, etas, and detailed fold results
        """
        dist_01 = None
        K = crossfit_data.K
        n1, d = crossfit_data.X_treatment.shape
        true_ATT = crossfit_data.true_ATT

        # Storage for results
        fold_results = []
        all_etas = []
        all_Y0_hat = np.zeros(n1)
        all_X_mapped = np.zeros((n1, d))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Cross-Fitting ATT Estimation")
            print(f"  K = {K} folds")
            print(f"  Strategy: {eta_strategy} (Metric={metric})")
            print(f"  n_treatment = {n1}, n_control = {len(crossfit_data.X_control)}")
            print(f"{'='*60}")
        
        # Iterate over folds
        for fold_idx in range(K):
            if verbose:
                print(f"\n--- Fold {fold_idx + 1}/{K} ---")
            
            # Get fold data
            fold_data = crossfit_data.get_fold_data(fold_idx)
            
            X_ctrl_train = fold_data['X_ctrl_train']
            X_trt_train = fold_data['X_trt_train']
            Y_ctrl_train = fold_data['Y_ctrl_train']
            Y_trt_train = fold_data['Y_trt_train']

            X_trt_test = fold_data['X_trt_test']
            Y_trt_test = fold_data['Y_trt_test']

            Y0_trt_train = fold_data.get('Y0_trt_train', None)
            Y0_trt_test = fold_data.get('Y0_trt_test', None)
            X0_train = fold_data.get('X0_train', None)
            X0_test = fold_data.get('X0_test', None)

            test_trt_idx = fold_data['test_trt_idx']
            transport_estimator = fold_data['transport_estimator']
            outcome_model = fold_data['outcome_model']
            
            # print("X_trt_test", X_trt_test.shape)
            # print("X0_test", X0_test.shape)
            # print("X_ctrl_train", X_ctrl_train.shape)

            n_test = len(X_trt_test)
            r = len(X_trt_train)/(len(X_trt_train) + len(X_ctrl_train))
            
            if verbose:
                print(f"  Training: {len(X_ctrl_train)} control")
                print(f"  Testing:  {n_test} treatment")
            
            # Check transport estimator is fitted
            if transport_estimator is None:
                raise ValueError(f"Transport estimator for fold {fold_idx} not fitted. "
                               "Call crossfit_data.fit_transport_maps() first.")
            
            # Apply eta strategy
            fold_etas = []
            
            if eta_strategy == 'global':
                if global_eta is not None:
                    eta = global_eta
                else:
                    # Find optimal global eta using training data
                    train_data = {
                        'X_treatment': X_trt_train,
                        'Y_treatment': Y_trt_train,
                        'Y0_treatment': Y0_trt_train,
                        'X_transported': X0_train,
                        'true_ATT': true_ATT
                    }
                    eta = self.find_best_eta_global(
                        train_data, outcome_model, eta_grid=None, metric=metric, verbose=False)
                
                # Map test treatment samples
                # X0_test = self.map_treatment_to_control(X_trt_test)
                X_mapped_test = eta * X0_test + (1 - eta) * X_trt_test
                fold_etas = [eta] * n_test
                
            elif eta_strategy == 'optimal':
                X_mapped_test = []
                # X0_test = self.map_treatment_to_control(X_trt_test)
                
                for i in range(n_test):
                    # print(f'i={i}')
                    x1 = X_trt_test[i]
                    x0 = X0_test[i]
                    
                    if individual_etas is not None:
                        eta = individual_etas[test_trt_idx[i]]
                        x_mapped = eta * x0 + (1 - eta) * x1
                    else:
                        # use only train data, as \eta_i X_i + (1-\eta_i) T(X_i) is independent samples given train data
                        eta, x_mapped = self.find_optimal_eta(x1, x0, X_ctrl_train)
                    
                    X_mapped_test.append(x_mapped)
                    fold_etas.append(eta)
                X_mapped_test = np.array(X_mapped_test).reshape((n_test, d))
                
            elif eta_strategy == None:
                X_mapped_test = X_trt_test
                fold_etas = [0] * n_test
                
            elif eta_strategy == 'oracle':
                # X0_test = self.map_treatment_to_control(X_trt_test)
                diff_X = (X0_train - X_trt_train)
                diff_Y = Y0_trt_train - outcome_model.predict(X_trt_train)
                
                if self.outcome_model_type == "linear":
                    beta, c = extract_linear_params(outcome_model)
                    gradf, _ = gradient_eval(func_type='linear', 
                                            X=X_trt_train, coef=beta, bias=c)
                
                if np.abs(diff_Y.mean()) < eps or np.abs((diff_X @ gradf).mean()) < eps:
                    eta = 0.
                else:
                    eta = diff_Y.mean() / (diff_X @ gradf).mean()
                # print(f"eta={eta: .3f}")

                    # eta = min(1, eta)
                    # eta = max(eta, 0)
                
                X_mapped_test = eta * X0_test + (1 - eta) * X_trt_test
                fold_etas = [eta] * n_test
            elif eta_strategy == 'ratio':

                if gaussian:
                    method = "gaussian"
                elif d <= 3:
                    method = "kde"
                else:
                    method = "ulsif"

                dual = MultidimRatioEstimator(
                    max_avg_weight, n_eta_grid=101, 
                    ot_estimator=transport_estimator, method=method, 
                    verbose=verbose)
                dual.fit(X_trt_train, X_ctrl_train, **kwargs)
                
                if not gaussian:
                    dist_01 = dual.avg_weights[10] #median
                else:
                    dist_01 = dual.avg_weights

                eta = 1. - dual.eta_star
                X_mapped_test = eta * X0_test + (1. - eta) * X_trt_test
                fold_etas = [eta] * n_test

            elif eta_strategy == 'ipm':
                mod = MultidimIPMEstimator(
                    max_avg_weight, n_eta_grid=101, 
                    ot_estimator=transport_estimator, metric=metric, 
                    verbose=verbose,
                    gaussian=gaussian
                    )
                mod.fit(X_trt_train, X_ctrl_train, **kwargs)
                
                eta = 1. - mod.eta_star
                if not gaussian:
                    dist_01 = mod.avg_weights[10] #median
                else:
                    dist_01 = mod.avg_weights

                X_mapped_test = eta * X0_test + (1. - eta) * X_trt_test     
                fold_etas = [eta] * n_test

            else:
                raise NotImplementedError("Use None, 'oracle', 'global', 'optimal'")
            
            
            # Predict counterfactual outcomes
            Y0_hat_test = outcome_model.predict(X_mapped_test)
            
            # Compute fold-specific ATT estimate
            ATT_fold = np.mean(Y_trt_test) - np.mean(Y0_hat_test)
            if true_ATT is not None:
                bias_fold = ATT_fold - true_ATT
            else:
                bias_fold = None

            # this only takes into account of the sample based estimation of expectation...
            variance_fold = (np.var(Y_trt_test, ddof=1)/(r*n_test) 
                            + np.var(Y0_hat_test, ddof=1)/((1-r)*n_test))
            
            # Store fold results
            fold_results.append({
                'ATT_estimate': ATT_fold,
                'bias': bias_fold,
                'dist_01': dist_01,
                'variance': variance_fold,
                'mean_eta': np.mean(fold_etas),
                'n_samples': n_test,
                'n_mapped': np.sum(np.array(fold_etas) > 0.01)
            })
            
            # Store individual predictions
            all_Y0_hat[test_trt_idx] = Y0_hat_test
            all_X_mapped[test_trt_idx] = X_mapped_test
            all_etas.extend(fold_etas)
            
            if verbose:
                print(f"  ATT estimate: {ATT_fold:.4f}")
                print(f"  Bias: {bias_fold}")
                print(f"  Mean η: {np.mean(fold_etas):.4f}")
        
        # Aggregate results across folds
        all_etas = np.array(all_etas)
        
        # Cross-fitted ATT estimate (average across folds)
        ATT_estimate = np.mean([fold['ATT_estimate'] for fold in fold_results])
        
        # Weighted average by sample size
        total_samples = sum(fold['n_samples'] for fold in fold_results)
        ATT_estimate_weighted = sum(fold['ATT_estimate'] * fold['n_samples'] 
                                   for fold in fold_results) / total_samples
        fold_variances = np.array([fold['variance'] for fold in fold_results])
        variance = np.sum(fold_variances) / (K**2)
        # MSE = Bias^2 + Variance


        if true_ATT is not None:
            bias = ATT_estimate - true_ATT
            mse = bias**2 + variance
        else:
            bias = None
            mse = None

        if crossfit_data.Y0_treatment is not None:
            # Overall prediction metrics
            prediction_error = np.mean((all_Y0_hat - crossfit_data.Y0_treatment) ** 2)
        else:
            prediction_error = None
        
        # variance = np.var(all_Y0_hat)
        if eta_strategy in ['ratio', 'ipm']:
            dist_01 = np.mean([fold.get('dist_01') for fold in fold_results])
        else:
            dist_01 = None
        
        results = {
            'ATT_estimate': ATT_estimate,
            'ATT_estimate_weighted': ATT_estimate_weighted,
            'true_ATT': true_ATT,
            'bias': bias,
            'variance': variance,
            'mse': mse,
            'etas': all_etas,
            'mean_eta': np.mean(all_etas),
            'X_mapped': all_X_mapped,
            'Y0_hat': all_Y0_hat,
            'Y0_true': crossfit_data.Y0_treatment,
            'prediction_MSE': prediction_error,
            'n_mapped': np.sum(all_etas > 0.01),
            'K': K,
            'dist_01': dist_01,
            'fold_results': fold_results,
            'fold_std': np.std([fold['ATT_estimate'] for fold in fold_results])
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Cross-Fitted Results (K={K}):")
            print(f"  True ATT: {true_ATT}")
            print(f"  Estimated ATT: {ATT_estimate:.4f}")
            print(f"  Bias: {bias}")
            print(f"  Std across folds: {results['fold_std']:.4f}")
            print(f"  Mean η: {np.mean(all_etas):.4f}")
            print(f"  Samples mapped (η > 0.01): {np.sum(all_etas > 0.01)} / {n1}")
            # print(f"  Prediction MSE: {prediction_error:.4f}")
            print(f"{'='*60}")
        
        return results


    def find_best_eta_global(self, data, outcome_model, eta_grid=None, verbose=True, metric="bias"):
        """Find best global η by minimizing absolute bias"""
        if eta_grid is None:
            eta_grid = np.linspace(0, 1, 101)
        n_eta = len(eta_grid)
        # biases = []
        # mses = []
        # variances = []
        # for eta in eta_grid:
        X_treatment = data['X_treatment']
        Y_treatment = data['Y_treatment']
        true_ATT = data['true_ATT']
        X0 = data['X_transported']
        Y0_true = data['Y0_treatment']
        n1,d = X_treatment.shape
        
        eta_exp = eta_grid.reshape(-1, 1, 1)
        X0_exp = X0.reshape(1, n1, d)
        X_trt_exp = X_treatment.reshape(1, n1, d)

        X_mapped_all = eta_exp * X0_exp + (1 - eta_exp) * X_trt_exp
        # Flatten and predict in ONE call
        X_mapped_flat = X_mapped_all.reshape(-1, d)
        Y0_hat_flat = outcome_model.predict(X_mapped_flat)
        Y0_hat_all = Y0_hat_flat.reshape(n_eta, n1)
        
        # ATT = E[Y(1) | A=1] - E[Y(0) | A=1]
        ATT_estimate = np.mean(Y_treatment) - np.mean(Y0_hat_all, axis=1)
        biases = ATT_estimate - true_ATT
        # print("biases", biases)
        # mses = np.mean((Y0_hat_all - Y0_true.reshape(1, -1)) ** 2, axis=1)
        # variances = np.var(Y0_hat_all, axis=1)

        # print("var same len", len(variances)==n_eta)
        # print("mse same len", len(mses)==n_eta)
        # print("bias same len", len(biases)==n_eta)
        
        if metric == "bias":
            best_idx = np.argmin(np.abs(biases))
        elif metric == "mse":
            best_idx = np.argmin(mses)
        elif metric == "variance":
            best_idx = np.argmin(variances)

        best_eta = eta_grid[best_idx]

        # plt.figure()
        # plt.plot(eta_grid, biases)
        # plt.axvline(x=best_eta, color='red')
        # plt.show()
        # plt.close()
        
        if verbose:
            print(f"Best global η: {best_eta:.3f} (|bias|={biases[best_idx]:.4f}, mse = {mses[best_idx]: .4f})")
        
        return best_eta
    
    # def compare_eta_strategies(self, data, eta_grid=None, verbose=False):
    #     """Compare different η strategies"""

    #     self.inverse_transport_plan = None
        
    #     results_comparison = {
    #         'eta_values': [],
    #         'ATT_estimates': [],
    #         'biases': [],
    #         'abs_biases': [],
    #         'prediction_MSEs': [],
    #         'mean_etas': []
    #     }
        
    #     # Try different global η values
    #     if eta_grid is None:
    #         eta_grid = np.linspace(0, 1, 41)
    #     for eta in eta_grid:
    #         results = self.estimate_ATT_with_eta(
    #             data, 
    #             eta_strategy='global', 
    #             global_eta=eta,
    #             verbose=verbose
    #         )
    #         # plt.figure()
    #         # plt.hist(data['X_control'], label="Control", alpha=0.5)
    #         # plt.hist(data['X_treatment'], label="Treatment", alpha=0.5)
    #         # plt.hist(results['X_mapped'], label="Mapped", alpha=0.5)
    #         # plt.title(f"eta={eta}")
    #         # plt.legend()
    #         # # plt.savefig(f"figs/ATT_mapped_eta{eta}.png")
    #         # plt.show()
    #         # plt.close()
    #         print(f"eta={eta}, bias={results['bias']}")
    #         results_comparison['eta_values'].append(eta)
    #         results_comparison['ATT_estimates'].append(results['ATT_estimate'])
    #         results_comparison['biases'].append(results['bias'])
    #         results_comparison['abs_biases'].append(np.abs(results['bias']))
    #         results_comparison['prediction_MSEs'].append(results['prediction_MSE'])
    #         results_comparison['mean_etas'].append(eta)
        
    #     # Find best global η
    #     best_idx = np.argmin(results_comparison['abs_biases'])
    #     best_global_eta = eta_grid[best_idx]
        
    #     # Get results at η=0 (no mapping)
    #     results_none = self.estimate_ATT_with_eta(data, eta_strategy=None, verbose=False)
        
    #     # Get results at best global η
    #     self.inverse_transport_plan = None
    #     results_best_global = self.estimate_ATT_with_eta(
    #         data, eta_strategy='global', global_eta=best_global_eta, verbose=False
    #     )
        
    #     # Reset and try optimal η strategy
    #     self.inverse_transport_plan = None
    #     results_optimal = self.estimate_ATT_with_eta(
    #         data,
    #         eta_strategy='optimal',
    #         verbose=False
    #     )
        
    #     results_comparison[None] = results_none
    #     results_comparison['best_global'] = results_best_global
    #     results_comparison['best_global_eta'] = best_global_eta
    #     results_comparison['optimal'] = results_optimal
        
    #     if verbose:
    #         print(f"\n{'='*60}")
    #         print("Strategy Comparison:")
    #         print(f"{'='*60}")
    #         print(f"No mapping (η=0):")
    #         print(f"  Bias: {results_none['bias']:>8.4f}, |Bias|: {np.abs(results_none['bias']):>8.4f}")
    #         print(f"\nBest global η={best_global_eta:.3f}:")
    #         print(f"  Bias: {results_best_global['bias']:>8.4f}, |Bias|: {np.abs(results_best_global['bias']):>8.4f}")
    #         print(f"\nIndividual optimal η (mean={results_optimal['mean_eta']:.3f}):")
    #         print(f"  Bias: {results_optimal['bias']:>8.4f}, |Bias|: {np.abs(results_optimal['bias']):>8.4f}")
        
    #     return results_comparison




def visualize_1d_mapping(generate_data, config=None, eta_strategies=None):
    """
    Visualize 1D case with histograms and prediction quality.
    
    Left plot: Histograms of control, treatment, and mapped distributions
    Right plot: Predicted vs. true Y(0) for treatment group
    """
    if config is None:
        config = {
            'rng': np.random.default_rng(42),
            'n0': 2000,
            'n1': 2000, # if n1 is 
            'd': 1,
            'outcome_type': 'quadratic',
            'quadratic_strength': 1.,
            'treatment_effect': 2.0,
            'overlap_param': -5.,
            'sigma': 0.1
        }
    
    if eta_strategies is None:
        eta_strategies = [None, 'mean_individual', 'individual', 'oracle']
    
    estimator = ATTEstimatorWithOT(method='exact', 
                                   outcome_model_type='linear',
                                   seed=42)
    data = generate_data(**config)
    
    # # Compute individual etas once
    # individual_etas = estimator.compute_all_individual_etas(
    #     data['X_control'], data['X_treatment'], verbose=False
    # )
    
    # Get results for each strategy
    results_dict = {}
    
    for strategy in eta_strategies:
        print(f"Start fitting strategy {strategy}")
        estimator.inverse_transport_plan = None
        
        if strategy == None:
            res = estimator.estimate_ATT_with_eta(
                data, eta_strategy=None, verbose=False
            )
        elif strategy == 'individual':
            res = estimator.estimate_ATT_with_eta(
                data, eta_strategy='optimal',
                individual_etas=None, verbose=False
            )
        elif strategy == 'mean_individual':
            res = estimator.estimate_ATT_with_eta(
                data, eta_strategy='global',
                global_eta=individual_etas.mean(), verbose=False
            )
        elif strategy == 'oracle':
            res = estimator.estimate_ATT_with_eta(
                data, eta_strategy='global', 
                global_eta=None, verbose=False
            )
        print(f"Estimated eta of {strategy}= {res['mean_eta']}")
        print(f"Estimated bias of {strategy}={res['bias']}")
        
        results_dict[strategy] = res
    
    # Create visualization
    n_strategies = len(eta_strategies)
    fig, axes = plt.subplots(n_strategies, 2, figsize=(14, 5*n_strategies))
    
    if n_strategies == 1:
        axes = axes.reshape(1, -1)
    
    X_control = data['X_control'].flatten()
    X_treatment = data['X_treatment'].flatten()
    
    
    for idx, strategy in enumerate(eta_strategies):
        res = results_dict[strategy]
        X_mapped = res['X_mapped'].flatten()
        Y0_true = res['Y0_true']
        Y0_hat = res['Y0_hat']
        
        # Left plot: Histograms
        ax = axes[idx, 0]
        
        bins = np.linspace(
            min(X_control.min(), X_treatment.min(), X_mapped.min()),
            max(X_control.max(), X_treatment.max(), X_mapped.max()),
            40
        )
        
        ax.hist(X_control, bins=bins, alpha=0.5, label='Control (A=0)',
                color='blue', edgecolor='black', density=True)
        ax.hist(X_treatment, bins=bins, alpha=0.5, label='Treatment (A=1)',
                color='orange', edgecolor='black', density=True)
        ax.hist(X_mapped, bins=bins, alpha=0.5, label='Mapped Treatment',
                color='green', edgecolor='black', density=True)
        
        # Mark convex hull boundaries
        ax.axvline(X_control.min(), color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Control Hull')
        ax.axvline(X_control.max(), color='red', linestyle='--', linewidth=2,
                  alpha=0.7)
        
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{strategy}\nη={res["mean_eta"]:.3f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Right plot: Predicted vs. True Y(0)
        ax = axes[idx, 1]
        
        ax.scatter(Y0_true, Y0_hat, alpha=0.6, s=30, color='green',
                  edgecolors='black', linewidth=0.5, label="predicted vs true")

        
        # Add regression line
        z = np.polyfit(Y0_true, Y0_hat, 1)
        p = np.poly1d(z)
        ax.plot(Y0_true, p(Y0_true), 'r-', linewidth=2, alpha=0.7,
               label=f'Y(0)|T=1 true (ytrue={z[0]:.2f}ypred+{z[1]:.2f})')
        
        ax.set_xlabel('True Y(0) | A=1', fontsize=11)
        ax.set_ylabel('Predicted Y(0) | A=1', fontsize=11)
        ax.set_title(f'Prediction Quality\n'
                    f'MSE={res["prediction_MSE"]:.3f}, '
                    f'Bias={res["bias"]:.3f}',
                    fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1d_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("1D COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'η':<8} {'MSE':<10} {'Bias':<10} {'|Bias|':<10}")
    print("-"*80)
    for strategy in eta_strategies:
        res = results_dict[strategy]
        print(f"{strategy:<20} {res['mean_eta']:<8.3f}"
              f"{res['prediction_MSE']:<10.3f} {res['bias']:<10.3f} {np.abs(res['bias']):<10.3f}")
    print("="*80)
    
    return results_dict



def visualize_2d_mapping(generate_data, config=None):

    if config is None:
        config = {
        'rng': np.random.default_rng(42),
        'n0': 500,
        'n1': 1000, # if n1 is 
        'd': 2,
        'outcome_type': 'quadratic',
        'quadratic_strength': 5.,
        'treatment_effect': 2.0,
        'overlap_param': 2.,
        'sigma': 0.01
        }


    estimator = ATTEstimatorWithOT(method='exact', outcome_model_type='linear', seed=42)
    data = generate_data(**config)

    # Get optimal mapping
    results_none = estimator.estimate_ATT_with_eta(data, eta_strategy=None, verbose=True)
    results_individual = estimator.estimate_ATT_with_eta(data, eta_strategy='optimal', verbose=True)
    results_global = estimator.estimate_ATT_with_eta(data, eta_strategy='global', global_eta=None, verbose=True)
    print("results_global", results_global)
    # None: results_none, 'individual':results_individual, 
    for strategy, results in {'oracle':results_global}.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Covariate space with mappings
        ax = axes[0]
        
        X_control = data['X_control']
        X_treatment = data['X_treatment']
        X_mapped = results['X_mapped']
        
        ax.scatter(X_control[:, 0], X_control[:, 1], 
                  alpha=0.6, s=50, color='blue', label='Control (A=0)')
        ax.scatter(X_treatment[:, 0], X_treatment[:, 1], 
                  alpha=0.6, s=50, color='orange', marker='^', label='Treatment (A=1)')
        ax.scatter(X_mapped[:, 0], X_mapped[:, 1], 
                  alpha=0.6, s=50, color='green', marker='s', label='Mapped Treatment')
        
        # Draw arrows for mapped samples
        for i in range(min(len(X_treatment), 150)):  # Limit arrows for clarity
            if results['etas'][i] > 0.01:
                ax.arrow(X_treatment[i, 0], X_treatment[i, 1],
                        X_mapped[i, 0] - X_treatment[i, 0],
                        X_mapped[i, 1] - X_treatment[i, 1],
                        head_width=0.1, head_length=0.1,
                        fc='gray', ec='gray', alpha=0.2, linewidth=0.5)
        
        # Convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(X_control)
        for simplex in hull.simplices:
            ax.plot(X_control[simplex, 0], X_control[simplex, 1], 
                   'r--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('X₁')
        ax.set_ylabel('X₂')
        ax.legend()
        ax.set_title(f'Covariate Mapping\nTrue ATT={data["true_ATT"]:.2f}, Est={results["ATT_estimate"]:.2f}, Bias={results["bias"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 2: Outcome predictions
        ax = axes[1]
        
        Y_treatment = data['Y_treatment']
        Y0_true = data['Y0_treatment']
        Y0_hat = results['Y0_hat']
        
        # Scatter of predicted vs true Y(0)
        ax.scatter(Y0_true, Y0_hat, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(Y0_true.min(), Y0_hat.min())
        max_val = max(Y0_true.max(), Y0_hat.max())
        
        ax.set_xlabel('True Y(0) for Treatment Group')
        ax.set_ylabel('Predicted Y(0) for Treatment Group')
        ax.set_title(f'Counterfactual Prediction Quality\nMSE={results["prediction_MSE"]:.3f}, eta={results["mean_eta"]: .3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'att_2d_visualization_{strategy}.png', dpi=150, bbox_inches='tight')
        plt.show()


def main(
    seed=42, r = 3/5, n = 5000, K = 5, 
    overlap_grid = np.linspace(0, -8, 10), 
    dist='norm', d=1, max_avg_weight=np.inf,
    outcome_model_type='linear',
    outcome_type='quadratic',
    path="results/normal_shift/results.csv",
    data_path="data/rhc_processed.csv",
    synthetic_path="data/rhc_nightmare.csv",
    metric='w1', **kwargs):
    
    ps_order = 1
    # overlap_grid = np.array([0.,-1.])
    # print("overlap", overlap_grid)
    eta_strategies = ['none', 'global(bias)', 'ratio', 'ipm']
    # eta_strategies = ['none', 'ratio', 'ipm']
    # eta_strategies = ['oracle', 'ratio', 'ipm']
    strategies = ['ipw', 'sipw', 'aipw', 'or', 'did']
    # strategies = []

    if dist == "rhc":
        data = load_rhc_data(
            path=data_path,
            standardize=False,
            semisynthetic=True,
            semisynthetic_path=synthetic_path
        )
        outcome_type = data['outcome_info']['type']
    elif dist == "lalonde":
        data = load_lalonde_data(
        path=data_path,
        version='psid', 
        standardize=True
        )
        outcome_type = data['outcome_info']['type']
    else:
        n0 = int(n*(1-r))
        config = {
            'rng': np.random.default_rng(seed),
            'n0': n0,
            'n1': int(n-n0), # if n1 is 
            'd': d,
            'outcome_type': outcome_type,
            # 'quadratic_strength': 1.,
            # 'treatment_effect': 2.0,
            'overlap_param': None,
            'sigma': 0.01
            }
        print(f"Dist type = {dist}, \n", config)
        outcome_type = config['outcome_type']

    print("Outcome type is ", outcome_type)

    if outcome_type == "binary":
        binary_outcome = True
    else:
        binary_outcome = False

    if outcome_type == "positive":
        use_log_link = True
    else:
        use_log_link = False

    results_sweep = {}
    max_avg_weight_ratio = max_avg_weight
    max_avg_weight_ipm = max_avg_weight

    for overlap_param in overlap_grid:
        print("\n" + "="*80)
        print(f"Start with delta={overlap_param}")
        print("="*80)

        if dist == "norm":
            # update delta
            config['overlap_param'] = overlap_param
            data = gen_norm_data(**config)
        elif dist == "beta":
            # update delta
            config['overlap_param'] = overlap_param
            data = gen_beta_data(**config)
        else:
            data = bootstrap(
                np.random.default_rng(int(overlap_param)), 
                data, 
                stratified=True)

        crossfit_data = CrossFitData(
            data['X_control'], data['X_treatment'], 
            data['Y_control'], data['Y_treatment'], 
            data['Y0_treatment'], data['true_ATT'], 
            K=K, random_state=seed,
            binary_outcome=binary_outcome,
            use_log_link=use_log_link
            )
        
        crossfit_data.fit_transport_maps(
            verbose=True, method='exact', oos_method='auto')

        crossfit_data.fit_outcome_models(
            outcome_model_type=outcome_model_type, 
            **kwargs)

        estimator = ATTEstimatorWithOT(seed=seed)
        
        # Get results for each strategy
        results_dict = {}
        
        for strategy in eta_strategies:
            print(f"Start fitting strategy {strategy}")
            # estimator.inverse_transport_plan = None
            
            if strategy == 'none':
                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy=None, verbose=False
                )
            elif strategy == 'individual':
                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy='optimal',
                    individual_etas=None, verbose=False
                )
            elif strategy == 'oracle':
                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy='oracle',
                    verbose=False
                )
            elif strategy == 'global(bias)':
                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy='global', metric='bias',
                    global_eta=None, verbose=False
                )
            elif strategy == 'global(mse)':
                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy='global', metric='mse',
                    global_eta=None, verbose=False
                )
            elif strategy == "ratio":
                if dist in ['norm', 'beta']:
                    if d == 20:
                        max_avg_weight_ratio = 3.5 
                    elif d == 5:
                        max_avg_weight_ratio = 1.1 
                elif dist == "lalonde":
                    max_avg_weight_ratio = 1e+07
                elif dist == "rhc":
                    max_avg_weight_ratio = 2.06112769e+00

                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy='ratio',
                    global_eta=None, verbose=True, 
                    max_avg_weight = max_avg_weight_ratio,
                    gaussian=True if dist == "norm" else False
                )
                # if overlap_param == 0 and dist != "norm":
                #     res["mean_eta"] = 0.25
                #     max_avg_weight_ratio = res.get('dist_01')
                # print("max_avg_weight_ratio", max_avg_weight_ratio)

            elif strategy == "ipm":
                if dist in ['norm', 'beta']:
                    if d == 20:
                        max_avg_weight_ipm = 20 
                    elif d == 5:
                        max_avg_weight_ipm = 2.5
                elif dist == "lalonde":
                    max_avg_weight_ipm = 1.5
                elif dist == "rhc":
                    max_avg_weight_ratio = 0.05969822

                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy='ipm',
                    global_eta=None, verbose=True, 
                    max_avg_weight = max_avg_weight_ipm,
                    metric=metric,
                    gaussian=True if dist == "norm" else False
                    )
                # if overlap_param == 0:
                #     res["mean_eta"] = 0.25
                #     max_avg_weight_ipm = res.get('dist_01')
                # print("max_avg_weight_ipm", max_avg_weight_ipm)

            elif strategy == "bias":
                res = estimator.estimate_ATT_with_eta_crossfit(
                    crossfit_data, eta_strategy='global',
                    global_eta=0.3, verbose=False
                )
            print(f"Estimated eta of {strategy}= {res['mean_eta']}")
            print(f"Estimated bias of {strategy}={res['bias']}")
            print(f"Estimated mse of {strategy}={res['prediction_MSE']}")
            
            results_dict[strategy] = res

        for strategy in strategies:
            res = estimate_ATT_crossfit(
                crossfit_data, method=strategy, ps_order=ps_order, verbose=True)
            res["mean_eta"] = None
            res['etas'] = None
            res['n_mapped'] = None

            print(f"Estimated bias of {strategy}={res['bias']}")
            print(f"Estimated mse of {strategy}={res['prediction_MSE']}")
            
            results_dict[strategy] = res

        results_sweep[overlap_param] = results_dict
    
    # expand the obtained results
    fields = ["mean_eta", "bias", "ATT_estimate"]

    flat_records = [
        {r"$\delta$": k1, "strategy": k2, **{f: v3.get(f) for f in fields}}
        for k1, v1 in results_sweep.items()
        for k2, v3 in v1.items()
    ]

    df_long = pd.DataFrame(flat_records)
    print(df_long)
    df_long.to_csv(path)

    # comparison to other estimators



def plot(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    markers = ['o', '*', '>', 'X', '^', 'P', 's', 'd', 'p', 'v', 'D', '<']

    for i, strategy in enumerate(df["strategy"].unique()):
        if strategy in ["oracle"]:
            lab = strategy + "(1st order)"
        else:
            lab = strategy
        subset = df[df["strategy"] == strategy]
        ax[0].plot(np.abs(subset[r"$\delta$"]), np.abs(subset["mean_eta"]), marker=markers[i], label=f"{lab}")

    ax[0].set_xlabel(r"|$\delta$|")
    ax[0].set_ylabel(r"$\bar{\eta}$")
    # plt.title(r"Bias comparison across overlap $\delta$")
    # ax[0].legend(title="Strategy", loc='upper left')

    for i, strategy in enumerate(df["strategy"].unique()):
        if strategy in ["oracle"]:
            lab = strategy + "(1st order)"
        else:
            lab = strategy
        subset = df[df["strategy"] == strategy]
        ax[1].plot(np.abs(subset[r"$\delta$"]), np.abs(subset["bias"]), marker=markers[i], label=f"{lab}")

    ax[1].set_xlabel(r"|$\delta$|")
    ax[1].set_ylabel("|Bias|")
    ax[1].set_yscale('log')
    # plt.title(r"Bias comparison across overlap $\delta$")
    # ax[1].legend(title="Strategy", loc='upper left')

    # for i, strategy in enumerate(df["strategy"].unique()):
    #     if strategy in ["oracle"]:
    #         lab = strategy + "(1st order)"
    #     else:
    #         lab = strategy
    #     subset = df[df["strategy"] == strategy]
    #     ax[2].plot(np.abs(subset[r"$\delta$"]), subset["variance"], marker=markers[i], label=f"{lab}")

    # ax[2].set_xlabel(r"|$\delta$|")
    # ax[2].set_ylabel("Variance")
    # ax[2].set_yscale('log')
    # # plt.title(r"Bias comparison across overlap $\delta$")
    # # ax[2].legend(title="Strategy", loc='upper left')

    # for i, strategy in enumerate(df["strategy"].unique()):
    #     if strategy in ["oracle"]:
    #         lab = strategy + "(1st order)"
    #     else:
    #         lab = strategy
    #     subset = df[df["strategy"] == strategy]
    #     ax[3].plot(np.abs(subset[r"$\delta$"]), subset["mse"], marker=markers[i], label=f"{lab}")

    # ax[3].set_xlabel(r"|$\delta$|")
    # ax[3].set_ylabel("MSE")
    # ax[3].set_yscale('log')
    # plt.title(r"Bias comparison across overlap $\delta$")
    # ax[3].legend(title="Strategy", loc='upper left')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=len(labels), frameon=True, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path)
    plt.show()
    plt.close()





if __name__ == "__main__":
    seed = 22
    # dist = 'norm'
    dist = 'lalonde'
    d = 20 # 20 # 5
    max_avg_weight = np.inf
    metric = "w1"
    outcome_model_type='polynomial'
    degree = 5
    outcome_type='quadratic'
    # overlap_grid = np.linspace(0,1,10)
    # main()

    # csv_path = f"results/{dist}_shift/{d}d_seed{seed}_{dist}_{outcome_type}_{outcome_model_type}.csv"
    csv_path = f"results/{dist}/seed{seed}_{dist}_{outcome_model_type}.csv"
    data_path="data/lalonde_psid.csv"
    main(seed=seed, r = 3/5, n = 10000, K = 4, 
        overlap_grid = np.linspace(0, 13, 14), d=d, dist=dist, max_avg_weight=max_avg_weight,
        outcome_model_type=outcome_model_type, outcome_type=outcome_type, path=csv_path,
        data_path=data_path,
        metric=metric, degree=degree)
    # main(seed=seed, r = 3/5, n = 10000, K = 5, 
    #     overlap_grid = np.array([10.]), d=d, dist=dist, max_avg_weight=max_avg_weight,
    #     outcome_model_type=outcome_model_type, outcome_type=outcome_type,
    #     metric=metric)
    # # load csv 
    df = pd.read_csv(csv_path)
    mean_results = df.groupby('strategy')[['mean_eta', 'bias', 'ATT_estimate']].apply(lambda x: x.abs().mean())
    var_results = df.groupby('strategy')[['mean_eta', 'bias', 'ATT_estimate']].var()
    print(mean_results)
    print(var_results)

    # plot(df, f"figs/{dist}/seed{seed}_{dist}_{outcome_model_type}.png")
    # all_results = comprehensive_comparison_experiment()
    # visualize_2d_mapping()
    # visualize_1d_mapping()
    # estimator = ATTEstimatorWithOT()
    # data = estimator.generate_data(n0=2000, n1=5000, d=2)  # Test with larger n0

    # # Run diagnostics to see what's failing
    # diagnostics = estimator.diagnose_eta_problems(
    #     data['X_control'], 
    #     data['X_treatment'],
    #     n_samples=20
    # )
    
