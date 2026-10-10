import ot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

class OptimalTransportEstimator:
    def __init__(self, method='sinkhorn', reg=1., oos_method='auto',
                 mapping_params=None):
        """
        Parameters:
        -----------
        method : str
            'emd' for exact transport or 'sinkhorn' for regularized
        reg : float
            Regularization parameter for sinkhorn
        oos_method : str
            Out-of-sample extension method:
            - 'auto': Uses built-in OT class behavior [DEFAULT]
            - 'joint_linear': Joint estimation with linear mapping
            - 'joint_kernel': Joint estimation with kernel mapping  
            - 'mapping_linear': MappingTransport with linear kernel
            - 'mapping_gaussian': MappingTransport with Gaussian kernel
            - 'linear_gaussian': LinearTransport (assumes Gaussian distributions)
        mapping_params : dict
            Additional parameters for mapping methods
        """
        self.method = method
        self.reg = reg
        self.oos_method = oos_method
        self.mapping_params = mapping_params or {}
        self.ot_estimator = None
        
        # For joint mapping methods
        self.L = None
        self.has_bias = False
        
    def estimate_inverse_transport(self, X_control, X_treatment, verbose=True):
        """Estimate inverse OT map: X|A=1 -> X|A=0"""
        d = X_control.shape[1]
        n1 = X_treatment.shape[0]
        n0 = X_control.shape[0]
        
        if verbose:
            print(f"Training samples: n_treatment={n1}, n_control={n0}")
        
        if self.oos_method == 'auto':
            if self.method == 'sinkhorn':
                self.ot_estimator = ot.da.SinkhornTransport(reg_e=self.reg, max_iter=100_000_000)
            else:
                self.ot_estimator = ot.da.EMDTransport(max_iter=100_000_000, metric="mahalanobis")
            self.ot_estimator.fit(Xs=X_treatment, Xt=X_control)
        
            
        # elif self.oos_method == 'joint_linear':
        #     default_params = {'reg': 1e-8, 'numItermax': 100, 'verbose': verbose,
        #                     'bias': True, 'log': False}
        #     params = {**default_params, **self.mapping_params}
        #     self.has_bias = params['bias']
            
        #     result = ot.da.joint_OT_mapping_linear(
        #         X_treatment, X_control,
        #         mu=self.reg,
        #         eta=params['reg'],
        #         bias=params['bias'],
        #         numItermax=params['numItermax'],
        #         verbose=params['verbose'],
        #         log=params['log']
        #     )
            
        #     if params['log']:
        #         coupling, self.L, log_dict = result
        #     else:
        #         coupling, self.L = result
            
        #     self.ot_estimator = coupling
            
        #     if verbose:
        #         print("Fitted joint linear mapping")
                
        # elif self.oos_method == 'joint_kernel':
        #     default_params = {'reg': 1e-8, 'eta': 1e-3, 'numItermax': 200, 
        #                     'kernel': 'gaussian', 'sigma': 1, 'verbose': verbose,
        #                     'bias': True, 'log': False}
        #     params = {**default_params, **self.mapping_params}
        #     self.has_bias = params['bias']
            
        #     self.X_treatment_train = X_treatment
        #     self.X_control_train = X_control
            
        #     result = ot.da.joint_OT_mapping_kernel(
        #         X_treatment, X_control,
        #         mu=self.reg,
        #         eta=params['eta'],
        #         bias=params['bias'],
        #         kerneltype=params['kernel'],
        #         sigma=params['sigma'],
        #         numItermax=params['numItermax'],
        #         verbose=params['verbose'],
        #         log=params['log']
        #     )
            
        #     if params['log']:
        #         coupling, self.L, log_dict = result
        #     else:
        #         coupling, self.L = result
            
        #     self.ot_estimator = coupling
            
        #     if verbose:
        #         print(f"Fitted joint {params['kernel']} kernel mapping")
                
        # elif self.oos_method == 'mapping_linear':
        #     default_params = {'mu': 1e0, 'eta': 1e-3, 'bias': True, 'tol': 1e-7, 
        #                     'max_iter': 200, 'verbose': verbose,
        #                     'max_inner_iter': 200, 'inner_tol':1e-6}
        #     params = {**default_params, **self.mapping_params}
            
        #     self.ot_estimator = ot.da.MappingTransport(kernel="linear", **params)
        #     self.ot_estimator.fit(Xs=X_treatment, Xt=X_control)
            
        #     if verbose:
        #         print("Fitted MappingTransport with linear kernel")
                
        # elif self.oos_method == 'mapping_gaussian':
        #     # default_params = {'mu': 1e-1, 'eta': 1e-5, 'bias': True, 
        #     #                 'sigma': 1, 'max_iter': 2000, 'verbose': verbose}
        #     # For speed (good balance)
        #     default_params = {
        #         'metric': 'sqeuclidean',
        #         'mu': 1e-0,           # Increase regularization (smoother, faster)
        #         'eta': 1e-3,         # Larger step size (10-100x faster!)
        #         'bias': True,
        #         'sigma': 1,
        #         'max_iter': 200,     # Reduce iterations (10x speedup)
        #         'verbose': verbose,
        #         'tol': 1e-5,          # Early stopping tolerance
        #         'max_inner_iter': 1000,
        #         'inner_tol':1e-6,
        #         # 'norm': 'max',
        #         'log': True
        #     }
        #     params = {**default_params, **self.mapping_params}
            
        #     self.ot_estimator = ot.da.MappingTransport(kernel="gaussian", **params)
        #     self.ot_estimator.fit(Xs=X_treatment, Xt=X_control)
            
        #     if verbose:
        #         print("Fitted MappingTransport with Gaussian kernel")
                
        # elif self.oos_method == 'linear_gaussian':
        #     self.ot_estimator = ot.da.LinearTransport()
        #     self.ot_estimator.fit(Xs=X_treatment, Xt=X_control)
            
        #     if verbose:
        #         print("Fitted LinearTransport (Gaussian assumption)")
        else:
            raise ValueError(f"Unknown oos_method: {self.oos_method}")
        
        return self
    
    def map_treatment_to_control(self, X_treatment):
        """Map treatment samples to control space."""
        if self.oos_method == 'joint_linear':
            if self.L is None:
                raise ValueError("Must call estimate_inverse_transport first")
            
            if self.has_bias:
                X_with_ones = np.hstack([X_treatment, np.ones((X_treatment.shape[0], 1))])
                return X_with_ones @ self.L
            else:
                return X_treatment @ self.L
            
        elif self.oos_method == 'joint_kernel':
            if self.L is None or not hasattr(self, 'X_treatment_train'):
                raise ValueError("Must call estimate_inverse_transport first")
            
            from ot.utils import kernel
            params = self.mapping_params
            K = kernel(X_treatment, self.X_treatment_train, 
                      method=params.get('kernel', 'gaussian'),
                      sigma=params.get('sigma', 1))
            
            if self.has_bias:
                K = np.hstack([K, np.ones((K.shape[0], 1))])
            
            return K @ self.L
        else:
            if self.ot_estimator is None:
                raise ValueError("Must call estimate_inverse_transport first")
            return self.ot_estimator.transform(Xs=X_treatment)


def plot_ot_comparison_1d():
    """Create a 1D example comparing different OT methods with in-sample vs out-of-sample"""
    np.random.seed(42)
    
    # Generate 1D data
    n_train = 800
    n_test = 500
    
    # Control: mixture of two Gaussians
    X_control_train = np.concatenate([
        np.random.normal(-2, 1., n_train),
        # np.random.normal(1, 0.8, n_train // 2)
    ]).reshape(-1, 1)
    
    # Treatment: shifted distribution
    X_treatment_train = np.concatenate([
        np.random.normal(0, 0.8, n_train)
        # np.random.normal(3, 0.7, n_train // 2)
    ]).reshape(-1, 1)
    
    # Test data (same distribution as treatment train)
    X_treatment_test = np.concatenate([
        np.random.normal(0, 0.7, n_test)
        # np.random.normal(3, 0.7, n_test // 2)
    ]).reshape(-1, 1)
    
    # Methods to compare
    methods = {
        'Auto (Barycentric+NN)': {'oos_method': 'auto'},
        'Joint Linear': {'oos_method': 'joint_linear', 'mapping_params': {'reg': 1e-6}},
        'Linear Gaussian': {'oos_method': 'linear_gaussian'},
        'Mapping Gaussian': {'oos_method': 'mapping_gaussian', 'mapping_params': {'sigma': 0.5}},
    }
    
    # Fit all methods and collect results
    results_test = {}
    results_train = {}
    w_distances = {}
    
    print("Fitting all methods...\n")
    for name, params in methods.items():
        print(f"Fitting {name}...")
        estimator = OptimalTransportEstimator(
            method='sinkhorn', 
            reg=0.1, 
            **params
        )
        
        # Fit on training data
        estimator.estimate_inverse_transport(X_control_train, X_treatment_train)
        
        # Map both train and test
        X_train_mapped = estimator.map_treatment_to_control(X_treatment_train).flatten()
        X_test_mapped = estimator.map_treatment_to_control(X_treatment_test).flatten()
        
        results_train[name] = X_train_mapped
        results_test[name] = X_test_mapped
        
        # Compute Wasserstein distance to target
        w_dist_train = wasserstein_distance(X_train_mapped, X_control_train.flatten())
        w_dist_test = wasserstein_distance(X_test_mapped, X_control_train.flatten())
        w_distances[name] = (w_dist_train, w_dist_test)
        
        print(f"  Train - Mean: {X_train_mapped.mean():.3f}, W-dist: {w_dist_train:.4f}")
        print(f"  Test  - Mean: {X_test_mapped.mean():.3f}, W-dist: {w_dist_test:.4f}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid: 4 methods x 2 columns (train/test) + 1 row for comparison bar chart
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, height_ratios=[1, 1, 0.8])
    
    # Common histogram parameters
    bins = 40
    range_vals = (-5, 5)
    alpha = 0.6
    
    # Plot each method (2 rows: train and test)
    for idx, (name, X_test_mapped) in enumerate(results_test.items()):
        X_train_mapped = results_train[name]
        
        # Training samples
        ax_train = fig.add_subplot(gs[0, idx])
        ax_train.hist(X_control_train.flatten(), bins=bins, range=range_vals, 
                     alpha=alpha, label='Target', color='blue', density=True)
        ax_train.hist(X_train_mapped, bins=bins, range=range_vals, 
                     alpha=0.8, label='Mapped (In-Sample)', color='green', 
                     density=True, histtype='step', linewidth=2)
        ax_train.axvline(X_control_train.mean(), color='blue', linestyle='--', 
                        linewidth=1.5, alpha=0.7)
        ax_train.axvline(X_train_mapped.mean(), color='green', linestyle='--', 
                        linewidth=1.5, alpha=0.7)
        ax_train.set_title(f'{name}\n(Training Data)', fontsize=10, fontweight='bold')
        ax_train.set_ylabel('Density')
        ax_train.legend(fontsize=8)
        ax_train.grid(True, alpha=0.3)
        ax_train.text(0.02, 0.98, f'W-dist: {w_distances[name][0]:.3f}', 
                     transform=ax_train.transAxes, va='top', fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Test samples
        ax_test = fig.add_subplot(gs[1, idx])
        ax_test.hist(X_control_train.flatten(), bins=bins, range=range_vals, 
                    alpha=alpha, label='Target', color='blue', density=True)
        ax_test.hist(X_treatment_test.flatten(), bins=bins, range=range_vals, 
                    alpha=0.4, label='Source', color='red', density=True)
        ax_test.hist(X_test_mapped, bins=bins, range=range_vals, 
                    alpha=0.8, label='Mapped (Out-of-Sample)', color='orange', 
                    density=True, histtype='step', linewidth=2)
        ax_test.axvline(X_control_train.mean(), color='blue', linestyle='--', 
                       linewidth=1.5, alpha=0.7)
        ax_test.axvline(X_test_mapped.mean(), color='orange', linestyle='--', 
                       linewidth=1.5, alpha=0.7)
        ax_test.set_title('(Test Data)', fontsize=10)
        ax_test.set_xlabel('Value')
        ax_test.set_ylabel('Density')
        ax_test.legend(fontsize=8)
        ax_test.grid(True, alpha=0.3)
        ax_test.text(0.02, 0.98, f'W-dist: {w_distances[name][1]:.3f}', 
                    transform=ax_test.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add comparison bar chart
    ax_compare = fig.add_subplot(gs[2, :])
    method_names = list(w_distances.keys())
    train_dists = [w_distances[m][0] for m in method_names]
    test_dists = [w_distances[m][1] for m in method_names]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax_compare.bar(x - width/2, train_dists, width, label='In-Sample (Train)', 
                           color='green', alpha=0.7)
    bars2 = ax_compare.bar(x + width/2, test_dists, width, label='Out-of-Sample (Test)', 
                           color='orange', alpha=0.7)
    
    ax_compare.set_xlabel('Method')
    ax_compare.set_ylabel('Wasserstein Distance to Target')
    ax_compare.set_title('Performance Comparison: In-Sample vs Out-of-Sample', 
                        fontweight='bold', fontsize=11)
    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels(method_names, rotation=15, ha='right')
    ax_compare.legend()
    ax_compare.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_compare.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Optimal Transport Methods Comparison: In-Sample vs Out-of-Sample Performance', 
                fontsize=13, fontweight='bold', y=0.995)
    
    plt.savefig('figs/ot_methods_comparison_detailed.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'ot_methods_comparison_detailed.png'")
    plt.show()
    
    # Print summary
    print("\n" + "=" * 90)
    print("Performance Summary (Wasserstein Distance to Target):")
    print("=" * 90)
    print(f"{'Method':<30} {'Train (In-Sample)':<20} {'Test (Out-of-Sample)':<20} {'Gap':<10}")
    print("-" * 90)
    for name in method_names:
        train_d, test_d = w_distances[name]
        gap = test_d - train_d
        print(f"{name:<30} {train_d:<20.4f} {test_d:<20.4f} {gap:<10.4f}")
    
    print("\n" + "=" * 90)
    print("Key Insights:")
    print("=" * 90)
    print("• Barycentric projection performs best IN-SAMPLE (training data)")
    print("• This is expected: it uses the exact OT coupling without approximation")
    print("• For OUT-OF-SAMPLE (test data), ALL methods use approximations")
    print("• The 'Auto' method uses nearest-neighbor for OOS, which can work well")
    print("• Parametric methods (joint linear/kernel) trade accuracy for:")
    print("  - Explicit functional form")
    print("  - Better generalization to far OOS points")
    print("  - Computational efficiency")
    print("• Kernel methods can be more flexible than linear for complex maps")


if __name__ == "__main__":
    plot_ot_comparison_1d()