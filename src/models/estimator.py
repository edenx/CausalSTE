import numpy as np
import ot
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Callable, Union, Literal
import warnings
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from .ot_maps import *

from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.stats import entropy

def _mmd(X, Y, gamma=1.0):
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

def _w1(X_train, X_trans):
    d = X_train.shape[1]
    # print(f"d={d}")
    if d <= 3:
        # Uniform weights
        a = np.ones(len(X_train)) / len(X_train)
        b = np.ones(len(X_trans)) / len(X_trans)
        M = ot.dist(X_train, X_trans, metric='euclidean')  # Cost matrix
        w1 = ot.emd2(a, b, M, numItermax=100_000_000)  # W1
        
    else:
        # w1 = ot.sliced.sliced_wasserstein_distance(
        #     X_train, X_trans, n_projections=300, p=1, seed=42)
        w1 = ot.sliced.sliced_wasserstein_distance(
            X_train, X_trans, n_projections=300, p=1, seed=42)
        # sliced 
    return w1

def _w2(X_train, X_trans):
    d = X_train.shape[1]
    # print(f"d={d}")
    if d <= 3:
    # Uniform weights
        a = np.ones(len(X_train)) / len(X_train)
        b = np.ones(len(X_trans)) / len(X_trans)
        M = ot.dist(X_train, X_trans, metric='mahalanobis')  # Cost matrix
        w2_squared = ot.emd2(a, b, M, numItermax=100_000_000)  # W2^2
        return np.sqrt(w2_squared)
    else:
        w2 = ot.sliced.sliced_wasserstein_distance(
            X_train, X_trans, n_projections=300, p=2, seed=42)
    return w2

def _chisq(method, ot_model, eta, X_test, X_train, **kwargs):
    if method == "kde":
        chisq =  _kde(ot_model, eta, X_test, X_train, **kwargs)
    elif method == "classifier":
        chisq = _classification(ot_model, eta, X_test, X_train, **kwargs)
    elif method == "ulsif":
        chisq = _ulsif(ot_model, eta, X_test, X_train)
    return chisq




def mahalanobis_sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    cov_estimator: Literal['empirical', 'shrinkage', 'ledoit_wolf'] = 'empirical',
    regularization: float = 1e-6,
    n_projections: int = 500,
    p: int = 2,
    seed: Optional[int] = None,
    use_library: bool = True
) -> float:
    """
    Compute Mahalanobis Sliced-Wasserstein distance between two distributions.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples_X, d)
        Samples from first distribution
    Y : np.ndarray, shape (n_samples_Y, d)
        Samples from second distribution
    covariates : np.ndarray, shape (n_covariates, d), optional
        Covariate array to estimate Mahalanobis metric.
        If None, uses pooled samples from X and Y.
    cov_estimator : str, default='empirical'
        Method to estimate covariance matrix:
        - 'empirical': Standard sample covariance
        - 'shrinkage': Shrinkage towards identity
        - 'ledoit_wolf': Ledoit-Wolf shrinkage estimator
    regularization : float, default=1e-6
        Regularization added to diagonal for numerical stability
    n_projections : int, default=500
        Number of random projections for sliced Wasserstein
    p : int, default=2
        Order of Wasserstein distance (typically 1 or 2)
    seed : int, optional
        Random seed for reproducibility
    use_library : bool, default=True
        If True, use clbonet's library (if available).
        If False, use standalone implementation.
    
    Returns
    -------
    distance : float
        Mahalanobis Sliced-Wasserstein distance
    
    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> Y = np.random.randn(100, 5) + 1
    >>> covariates = np.random.randn(200, 5)
    >>> dist = mahalanobis_sliced_wasserstein_distance(X, Y, covariates)
    >>> print(f"Distance: {dist:.4f}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs
    assert X.ndim == 2 and Y.ndim == 2, "X and Y must be 2D arrays"
    assert X.shape[1] == Y.shape[1], "X and Y must have same number of features"
    d = X.shape[1]
    
    # Determine covariates for covariance estimation
    if covariates is None:
        covariates = np.vstack([X, Y])
    else:
        assert covariates.shape[1] == d, "Covariates must have same dimension as X and Y"
    
    # Estimate covariance matrix
    cov_matrix = _estimate_covariance(covariates, method=cov_estimator, regularization=regularization)
    
    # Compute distance
    if use_library:
        try:
            return _compute_with_library(X, Y, cov_matrix, n_projections, p, seed)
        except ImportError:
            print("Library not available, falling back to standalone implementation")
            return _compute_standalone(X, Y, cov_matrix, n_projections, p)
    else:
        return _compute_standalone(X, Y, cov_matrix, n_projections, p)


def _estimate_covariance(
    data: np.ndarray,
    method: str = 'empirical',
    regularization: float = 1e-6
) -> np.ndarray:
    """Estimate covariance matrix with different methods."""
    n, d = data.shape
    
    if method == 'empirical':
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        cov = (data_centered.T @ data_centered) / (n - 1)
        
    elif method == 'shrinkage':
        # Shrinkage towards identity
        data_centered = data - np.mean(data, axis=0)
        sample_cov = (data_centered.T @ data_centered) / (n - 1)
        
        # Optimal shrinkage parameter (simple Ledoit-Wolf style)
        trace = np.trace(sample_cov)
        mu = trace / d
        delta = np.linalg.norm(sample_cov - mu * np.eye(d), 'fro')**2 / d
        
        # Frobenius norm of sample covariance
        beta = 0.0
        for i in range(n):
            centered = data_centered[i:i+1, :]
            term = centered.T @ centered - sample_cov
            beta += np.linalg.norm(term, 'fro')**2
        beta = beta / (n**2)
        
        # Shrinkage intensity
        kappa = beta / delta if delta > 0 else 0
        shrinkage = max(0, min(1, kappa))
        
        cov = shrinkage * mu * np.eye(d) + (1 - shrinkage) * sample_cov
        
    elif method == 'ledoit_wolf':
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov = lw.fit(data).covariance_
        except ImportError:
            print("scikit-learn not available, using empirical covariance")
            return _estimate_covariance(data, 'empirical', regularization)
    else:
        raise ValueError(f"Unknown covariance estimator: {method}")
    
    # Add regularization for numerical stability
    cov += regularization * np.eye(d)
    
    return cov


def _compute_with_library(
    X: np.ndarray,
    Y: np.ndarray,
    cov_matrix: np.ndarray,
    n_projections: int,
    p: int,
    seed: Optional[int]
) -> float:
    """Compute using clbonet's library."""
    try:
        import torch
        from hswfs.sw import sliced_wasserstein
        from hswfs.manifold.mahalanobis import Mahalanobis
    except ImportError:
        raise ImportError(
            "Could not import required libraries. Install with:\n"
            "git clone https://github.com/clbonet/Sliced-Wasserstein_Distances_and_Flows_on_Cartan-Hadamard_Manifolds\n"
            "cd Sliced-Wasserstein_Distances_and_Flows_on_Cartan-Hadamard_Manifolds\n"
            "python setup.py install"
        )
    
    # Convert to torch tensors
    X_torch = torch.from_numpy(X).float()
    Y_torch = torch.from_numpy(Y).float()
    A_torch = torch.from_numpy(cov_matrix).float()
    
    # Create Mahalanobis manifold
    d = X.shape[1]
    manifold = Mahalanobis(d, A_torch)
    
    # Compute distance
    distance = sliced_wasserstein(
        X_torch, 
        Y_torch, 
        n_projections, 
        manifold,
        p=p,
        seed=seed
    )
    
    return float(distance.item())


def _compute_standalone(
    X: np.ndarray,
    Y: np.ndarray,
    cov_matrix: np.ndarray,
    n_projections: int,
    p: int
) -> float:
    """Standalone implementation without external libraries."""
    from scipy import linalg
    
    # Compute matrix square root: A^(1/2)
    A_sqrt = linalg.sqrtm(cov_matrix).real
    
    # Transform data
    X_transformed = X @ A_sqrt.T
    Y_transformed = Y @ A_sqrt.T
    
    # Compute sliced Wasserstein on transformed data
    d = X.shape[1]
    distances = []
    
    for _ in range(n_projections):
        # Random direction on unit sphere
        theta = np.random.randn(d)
        theta = theta / np.linalg.norm(theta)
        
        # Project samples
        X_proj = X_transformed @ theta
        Y_proj = Y_transformed @ theta
        
        # Sort projections
        X_sorted = np.sort(X_proj)
        Y_sorted = np.sort(Y_proj)
        
        # Compute 1D Wasserstein distance
        if p == 1:
            dist = np.mean(np.abs(X_sorted - Y_sorted))
        elif p == 2:
            dist = np.sqrt(np.mean((X_sorted - Y_sorted) ** 2))
        else:
            dist = np.mean(np.abs(X_sorted - Y_sorted) ** p) ** (1/p)
        
        distances.append(dist ** p)
    
    return np.mean(distances) ** (1/p)




def _classification(ot_model, eta: float, X_test: np.ndarray, 
                              Y_train: np.ndarray, classifier_type="logistic") -> float:
    """
    Estimate E[w_η] using calibrated classification.
    
    Key improvements:
    - Use calibrated classifier
    - Balanced sampling
    - Cross-validation
    - Better numerical stability
    """
    # Transport test samples
    X_trans_full = ot_model.map_treatment_to_control(X_test)
    X_trans = eta * X_test + (1 - eta) * X_trans_full
    
    # Balance sample sizes to avoid bias
    n_train = len(Y_train)
    n_test = len(X_trans)
    n_min = min(n_train, n_test)
    
    # Random subsample to balance
    idx_train = np.random.choice(n_train, n_min, replace=False)
    idx_test = np.random.choice(n_test, n_min, replace=False)
    
    Y_train_bal = Y_train[idx_train]
    X_trans_bal = X_trans[idx_test]
    
    # Create classification dataset (BALANCED)
    X_combined = np.vstack([Y_train_bal, X_trans_bal])
    y_combined = np.hstack([np.ones(n_min), np.zeros(n_min)])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    Y_train_scaled = scaler.transform(Y_train)  # Transform FULL training set
    
    # Split for calibration
    X_train_clf, X_cal, y_train_clf, y_cal = train_test_split(
        X_scaled, y_combined, test_size=0.3, random_state=42, stratify=y_combined
    )
    
    # Train base classifier
    if classifier_type == 'logistic':
        base_clf = LogisticRegressionCV(cv=3, max_iter=1e7, random_state=42)
    elif classifier_type == 'mlp':
        base_clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500,
                                 random_state=42, early_stopping=True)
    else:
        raise ValueError(f"Unknown classifier: {self.classifier_type}")
    
    base_clf.fit(X_train_clf, y_train_clf)
    
    # Calibrate probabilities using held-out data
    calibrated_clf = CalibratedClassifierCV(
        base_clf, method='sigmoid', cv='prefit'
    )
    calibrated_clf.fit(X_cal, y_cal)
    
    # Predict on FULL training set
    probs = calibrated_clf.predict_proba(Y_train_scaled)[:, 1]  # P(class=1|y)
    
    # Compute density ratio with smoothing
    # w(y) = h(y) / (1 - h(y))
    # Use additive smoothing to avoid division by zero
    epsilon = 0.01
    probs_smoothed = np.clip(probs, epsilon, 1 - epsilon)
    
    weights = probs_smoothed / (1 - probs_smoothed)
    
    # Remove outliers (top 2%)
    weights_trimmed = weights[weights < np.percentile(weights, 98)]
    
    return np.mean(weights_trimmed)

def _ulsif(ot_model, eta: float, X_test: np.ndarray,
                               Y_train: np.ndarray) -> float:
    """
    uLSIF: Unconstrained Least-Squares Importance Fitting (FIXED VERSION).
    
    Key fixes:
    - Better bandwidth selection
    - Use training data for centers (more stable)
    - Add numerical safeguards
    """
    # Transport samples
    X_trans_full = ot_model.map_treatment_to_control(X_test)
    X_trans = eta * X_test + (1 - eta) * X_trans_full
    
    # Use TRAINING samples as centers (more stable than using transported)
    n_centers = min(100, len(Y_train) // 5)
    idx_centers = np.random.choice(len(Y_train), n_centers, replace=False)
    centers = Y_train[idx_centers]
    
    # Better bandwidth: use cross-validation or Scott's rule
    # Scott's rule for RBF: σ = n^{-1/(d+4)} * std
    d = X_test.shape[1]
    n = len(Y_train)
    
    # Compute bandwidth from training data variance
    std_train = np.std(Y_train, axis=0).mean()
    sigma = std_train * (n ** (-1.0 / (d + 4)))
    
    # Ensure reasonable bandwidth
    sigma = max(sigma, 0.1)
    sigma = min(sigma, 10.0)
    
    # RBF kernel
    def kernel_matrix(X, centers):
        from scipy.spatial.distance import cdist
        dists_sq = cdist(X, centers, 'sqeuclidean')
        return np.exp(-dists_sq / (2 * sigma**2))
    
    # Compute H = E_η[K(X,c)K(X,c)ᵀ]
    K_eta = kernel_matrix(X_trans, centers)
    H = (K_eta.T @ K_eta) / len(X_trans)
    
    # Compute h = E_train[K(Y,c)]
    K_train = kernel_matrix(Y_train, centers)
    h = np.mean(K_train, axis=0)
    
    # Regularization (important for stability!)
    lambda_reg = max(0.01, 1e-3 * np.trace(H) / n_centers)
    
    # Solve for α
    try:
        alpha = np.linalg.solve(H + lambda_reg * np.eye(n_centers), h)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(H + lambda_reg * np.eye(n_centers), h, rcond=None)[0]
    
    # Project to non-negative (IMPORTANT for valid density ratios!)
    alpha = np.maximum(alpha, 0)
    
    # Normalize to ensure E_η[w] ≈ 1 (optional but helps)
    w_eta_samples = K_eta @ alpha
    if np.mean(w_eta_samples) > 0:
        alpha = alpha / np.mean(w_eta_samples)
    
    # Evaluate on training samples
    w_train = K_train @ alpha
    
    # Clip extreme values
    # w_train_clipped = np.clip(w_train, 0, np.percentile(w_train, 99))
    w_train_clipped = w_train
    
    return np.mean(w_train_clipped)


def _kde(ot_model, eta, X_test, X_train, kde_bandwidth=None):
    dimension = X_test.shape[1]
        
    if dimension > 4:
        warnings.warn(
            f"KDE unreliable for d={dimension}>4. "
            "Consider using dual formulation instead."
        )
    
    # Determine bandwidth
    if kde_bandwidth is None:
        # Scott's rule scaled for dimension
        bandwidth = len(X_train) ** (-1./(dimension + 4))
    else:
        bandwidth = kde_bandwidth

    # Fit KDE on training data (fixed)
    kde_train = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_train.fit(X_train)

    # Transport test samples
    X_trans_full = ot_model.map_treatment_to_control(X_test)
    X_trans = eta * X_test + (1 - eta) * X_trans_full

    # Fit KDE on transported data
    kde_eta = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_eta.fit(X_trans)

    # Evaluate on training samples
    log_p_train = kde_train.score_samples(X_train)
    log_p_eta = kde_eta.score_samples(X_train)

    # Density ratio with clipping for numerical stability
    log_ratio = np.clip(log_p_train - log_p_eta, -20, 20)
    weights = np.exp(log_ratio)

    # Clip outliers
    weights_clipped = np.clip(weights, 0, np.percentile(weights, 99))

    return np.mean(weights_clipped)
    

def _fit_gaussian(max_avg_weight, X_test, X_train, verbose=False):
    """
    Gaussian baseline with robust high-dimensional covariance estimation.
    
    Implements multiple strategies:
    1. Ledoit-Wolf shrinkage (default, best for general case)
    2. Diagonal approximation (fast, works well when features are ~independent)
    3. Ridge regularization (manual control)
    4. Oracle approximation (very high-d)
    """
    
    mu_test = np.mean(X_test, axis=0)
    mu_train = np.mean(X_train, axis=0)
    n_test, n_train = len(X_test), len(X_train)
    d = X_test.shape[1]
    
    # Compute sample covariances
    cov_test = np.cov(X_test.T, bias=False)
    cov_train = np.cov(X_train.T, bias=False)
    
    # Pooled covariance
    Sigma_sample = (cov_test * (n_test - 1) + cov_train * (n_train - 1)) / (n_test + n_train - 2)
    
    # Handle 1D case
    if d == 1:
        if np.isscalar(Sigma_sample):
            Sigma_sample = np.array([[Sigma_sample]])
        Sigma = Sigma_sample + 1e-8  # Small numerical stability
    else:
        # High-dimensional case: choose estimation strategy
        n_total = n_test + n_train
        ratio = d / n_total
        condition_number = np.linalg.cond(Sigma_sample)
        
        if verbose:
            print(f"\n  Covariance estimation diagnostics:")
            print(f"    Dimension d = {d}")
            print(f"    Total samples n = {n_total}")
            print(f"    Ratio d/n = {ratio:.3f}")
            print(f"    Condition number = {condition_number:.2e}")
        

        # Auto-select based on problem characteristics
        if ratio > 0.8:  # Very high-dimensional
            method = 'diagonal'
        elif ratio > 0.3 or condition_number > 1e6:  # Moderately ill-conditioned
            method = 'ledoit_wolf'
        else:  # Well-conditioned
            method = 'sample'
        
        if verbose:
            print(f"    Selected method: {method}")
        
        # Estimate Sigma using selected method
        if method == 'sample':
            # Standard sample covariance with small ridge
            Sigma = Sigma_sample + 1e-6 * np.eye(d)
            
        elif method == 'ledoit_wolf':
            # Ledoit-Wolf shrinkage: optimal linear combination
            # of sample covariance and diagonal
            Sigma = _ledoit_wolf_shrinkage(X_test, X_train)
            
        elif method == 'diagonal':
            # Use diagonal covariance (independence assumption)
            Sigma = np.diag(np.diag(Sigma_sample)) + 1e-8 * np.eye(d)
            
        elif method == 'ridge':
            # Manual ridge regularization
            ridge_param = 0.1
            trace_sigma = np.trace(Sigma_sample) / d
            Sigma = Sigma_sample + ridge_param * trace_sigma * np.eye(d)
            
        elif method == 'graphical_lasso':
            # Sparse inverse covariance (requires sklearn)
            try:
                from sklearn.covariance import GraphicalLassoCV
                # Combine data for estimation
                X_combined = np.vstack([X_test, X_train])
                gl = GraphicalLassoCV(cv=3, max_iter=100)
                gl.fit(X_combined)
                Sigma = gl.covariance_
            except ImportError:
                if verbose:
                    print("    GraphicalLasso not available, falling back to Ledoit-Wolf")
                Sigma = _ledoit_wolf_shrinkage(X_test, X_train)
                
        else:
            raise ValueError(f"Unknown covariance estimator: {method}")
        if verbose:
            cond_final = np.linalg.cond(Sigma)
            print(f"    Final condition number: {cond_final:.2e}")
    
    # Compute Mahalanobis distance
    delta = mu_train - mu_test
    
    # Robust computation of Delta_sq
    try:
        # Method 1: Try Cholesky decomposition (most stable)
        L = np.linalg.cholesky(Sigma)
        z = np.linalg.solve(L, delta)
        Delta_sq = np.dot(z, z)
        
        if verbose:
            print(f"    Used Cholesky decomposition for Δ²")
            
    except np.linalg.LinAlgError:
        # Method 2: Fall back to pseudo-inverse if Cholesky fails
        if verbose:
            print(f"    Cholesky failed, using pseudo-inverse")
        
        Sigma_inv = np.linalg.pinv(Sigma, rcond=1e-10)
        Delta_sq = delta @ Sigma_inv @ delta
    
    # Sanity check
    if Delta_sq < 0:
        if verbose:
            print(f"    Warning: Negative Δ² = {Delta_sq:.6f}, setting to 0")
        Delta_sq = 0
    
    if Delta_sq < 1e-10:
        if verbose:
            print(f"    Warning: Very small Δ² ≈ 0, distributions nearly identical")
        eta_star = 1.0
        avg_weights = 1.0
        Delta_sq = Delta_sq
        Sigma = Sigma
        return eta_star, avg_weights
    
    # Compute optimal eta
    eta = np.sqrt(np.log(max_avg_weight) / Delta_sq)
    eta = np.clip(eta, 0, 1)
    w = np.exp(eta**2 * Delta_sq)
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Δ² = {Delta_sq:.4f}")
        print(f"    η* = {eta:.4f}")
        print(f"    E[w] = {w:.2f}")
    
    # Store results
    eta_star = eta
    avg_weights = w
    Delta_sq = Delta_sq
    Sigma = Sigma
    
    return eta_star, avg_weights

def _ledoit_wolf_shrinkage(self, X_test, X_train):
    """
    Ledoit-Wolf optimal shrinkage estimator.
    
    Shrinks sample covariance toward diagonal (scaled identity):
    Σ_shrunk = (1-α) * Σ_sample + α * μ * I
    
    where α is chosen to minimize MSE.
    """
    
    # Combine samples for joint estimation
    X = np.vstack([X_test, X_train])
    n, d = X.shape
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Sample covariance
    S = (X_centered.T @ X_centered) / n
    
    # Target: scaled identity
    mu = np.trace(S) / d
    F = mu * np.eye(d)
    
    # Compute optimal shrinkage intensity
    # Based on Ledoit-Wolf (2004)
    
    # Compute shrinkage numerator
    X_sq = X_centered ** 2
    phi_mat = (X_sq.T @ X_sq) / n - S ** 2
    phi = np.sum(phi_mat)
    
    # Compute shrinkage denominator  
    gamma = np.linalg.norm(S - F, 'fro') ** 2
    
    # Optimal shrinkage
    if gamma > 1e-10:
        alpha = max(0, min(phi / gamma, 1))
    else:
        alpha = 1.0  # If S ≈ F, use target
    
    # Shrunk covariance
    Sigma_shrunk = (1 - alpha) * S + alpha * F
    
    if self.verbose:
        print(f"    Ledoit-Wolf shrinkage intensity α = {alpha:.4f}")
        print(f"    (α=0: use sample cov, α=1: use scaled identity)")
    
    return Sigma_shrunk

class MultidimRatioEstimator:
    """
    Multidimensional plug-in KDE method for η estimation.
    
    Works well for low dimensions (d ≤ 3-4).
    Estimates E[p_train(Y) / p_η(Y)] directly using KDE.
    """
    
    def __init__(self, 
                 max_avg_weight: float = 10.0,
                 n_eta_grid: int = 41,
                 # ot_reg: float = 0.1,
                 ot_estimator=None,
                 # kde_bandwidth: float = None,
                 verbose=False,
                 method='kde'):
        self.max_avg_weight = max_avg_weight
        # print(self.max_avg_weight)
        self.n_eta_grid = n_eta_grid
        self.ot_model = ot_estimator
        self.method = method
        
        self.verbose = verbose

        self.eta_star = None
        self.dimension = None
        self.eta_grid = None
        self.avg_weights = None

    def fit(self, X_test, X_train, **kwargs):

        if self.method == "gaussian":
            self._fit_gaussian(X_test, X_train)
        else:
            self._fit(X_test, X_train, **kwargs)

        return self    

    def _fit_gaussian(self, X_test, X_train):
        """Gaussian baseline (same covariates)."""
        mu_test = np.mean(X_test, axis=0)
        mu_train = np.mean(X_train, axis=0)

        n_test, n_train = len(X_test), len(X_train)
        cov_test = np.cov(X_test.T)
        cov_train = np.cov(X_train.T)

        d = X_test.shape[1]
        if d == 1:
            Sigma = (cov_test * n_test + cov_train * n_train) / (n_test + n_train)
            if np.isscalar(Sigma):
                Sigma = np.array([[Sigma]])
        else:
            Sigma = (cov_test * n_test + cov_train * n_train) / (n_test + n_train)
            Sigma += 1e-6 * np.eye(d)

        delta = mu_train - mu_test
        Delta_sq = delta @ np.linalg.solve(Sigma, delta)

        if Delta_sq < 1e-10:
            return 1.0, 1.0

        eta = np.sqrt(np.log(self.max_avg_weight) / Delta_sq)
        eta = np.clip(eta, 0, 1)
        w = np.exp(eta**2 * Delta_sq)

        if self.verbose:
            print(f"  Δ² = {Delta_sq:.4f}, η* = {eta:.4f}, E[w] = {w:.2f}")
        
        self.eta_star = eta
        self.avg_weights = w
        return self
        
    def _fit(self, X_test: np.ndarray, X_train: np.ndarray, **kwargs):
        """Fit using multidimensional KDE."""
        X_test = np.atleast_2d(X_test)
        X_train = np.atleast_2d(X_train)
        
        self.dimension = X_test.shape[1]
        
        # Grid search
        self.eta_grid = np.linspace(0, 1, self.n_eta_grid)
        # self.eta_grid = eta_grid_dense_at_zero(self.n_eta_grid, power=2)
        self.avg_weights = []
        
        
        for i, eta in enumerate(self.eta_grid):
            if self.method == "kde":
                avg_weight =  _kde(self.ot_model, eta, X_test, X_train, **kwargs)
            elif self.method == "classifier":
                avg_weight = _classification(self.ot_model, eta, X_test, X_train, **kwargs)
            elif self.method == "ulsif":
                avg_weight = _ulsif(self.ot_model, eta, X_test, X_train)

            self.avg_weights.append(avg_weight)
        
        self.avg_weights = np.array(self.avg_weights)
        # self.max_avg_weight = np.quantile(self.avg_weights, q=q)
        
        # Find optimal η
        feasible = self.avg_weights <= self.max_avg_weight
        if np.any(feasible):
            self.eta_star = self.eta_grid[feasible].max()
        else:
            warnings.warn(f"No feasible η, min weight: {self.avg_weights.min():.2f}")
            self.eta_star = self.eta_grid[np.argmin(self.avg_weights)]
        
        idx_star = np.argmin(np.abs(self.eta_grid - self.eta_star))
        
        if self.verbose:
            print(f"  Optimal η* = {self.eta_star:.4f}")
            print(f"  E[w_η*] = {self.avg_weights[idx_star]:.2f}")
            print(self.avg_weights)
        return self
    
    def transform(self, X_test: np.ndarray, eta: Optional[float] = None):
        """Apply geodesic transport."""
        if eta is None:
            eta = self.eta_star
        
        X_trans_full = self.ot_model.map_treatment_to_control(X_test)
        return eta * X_test + (1 - eta) * X_trans_full


def eta_grid_dense_at_zero(n_points=50, power=2):
    """
    Create eta grid with more density near 0 using power transformation.
    
    Parameters:
    -----------
    n_points : int
        Number of grid points
    power : float
        Power for transformation (higher = more density near 0)
        - power=1: uniform
        - power=2: quadratic (moderate density)
        - power=3: cubic (high density)
    """
    t = np.linspace(0, 1, n_points)
    eta = t ** power
    return eta

class HellingerEstimator:
    """
    Pre-compute densities and enable efficient Hellinger evaluation for any eta.
    """
    
    def __init__(self, ot_model, X_test, X_train, kde_bandwidth=None, method='auto'):
        """
        Pre-compute and store information needed for Hellinger distance.
        
        Parameters:
        -----------
        ot_model : object
            Trained optimal transport model with map_treatment_to_control method
        X_test : array-like, shape (n_test, d)
            Samples from P_X^1 (target distribution)
        X_train : array-like, shape (n_train, d)
            Samples from P_X^0 (source distribution)
        kde_bandwidth : float or None
            Bandwidth for KDE. If None, uses Scott's rule.
        method : str
            'auto', 'kde', 'knn', or 'sliced'
        """
        
        self.X_train = X_train
        self.X_test = X_test
        self.dimension = X_train.shape[1]
        self.n_train = len(X_train)
        self.method = method
        
        # Auto-select method
        if self.method == 'auto':
            if self.dimension <= 3:
                self.method = 'kde'
            elif self.dimension <= 10:
                self.method = 'knn'
            else:
                self.method = 'sliced'
            # print(f"Auto-selected method: {self.method} for d={self.dimension}")
        
        # Pre-compute transported samples (do this once!)
        # print("Pre-computing transported samples...")
        self.X_transported_full = ot_model.map_treatment_to_control(X_test)
        
        # Determine bandwidth
        if kde_bandwidth is None:
            self.bandwidth = self.n_train ** (-1./(self.dimension + 4))
        else:
            self.bandwidth = kde_bandwidth
        
        # Pre-fit KDE on P_0 (source) if using KDE method
        if self.method == 'kde':
            if self.dimension > 4:
                warnings.warn(
                    f"KDE unreliable for d={self.dimension}>4. "
                    "Consider method='knn' or method='sliced'."
                )
            
            # print(f"Fitting KDE on P_0 with bandwidth={self.bandwidth:.4f}...")
            self.kde_P0 = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
            self.kde_P0.fit(X_train)
        else:
            self.kde_P0 = None
        
        # print("✓ Initialization complete. Ready to evaluate Hellinger for any eta.")
    
    def evaluate(self, eta, n_projections=50, k_neighbors=5):
        """
        Evaluate Hellinger distance H(P_0, P_eta) for a given eta.
        
        Parameters:
        -----------
        eta : float or array-like
            Interpolation parameter(s) in [0, 1]
        n_projections : int
            Number of projections for 'sliced' method
        k_neighbors : int
            Number of neighbors for 'knn' method
            
        Returns:
        --------
        hellinger_distance : float or array
            Hellinger distance(s)
        """
        
        # Handle multiple eta values
        if isinstance(eta, (list, np.ndarray)):
            return np.array([self.evaluate(e, n_projections, k_neighbors) for e in eta])
        
        # Generate samples from P_eta (this is cheap!)
        X_eta = self._generate_P_eta_samples(eta)
        
        # Compute Hellinger using selected method
        if self.method == 'kde':
            return self._hellinger_kde(X_eta)
        elif self.method == 'knn':
            return self._hellinger_knn(X_eta, k_neighbors)
        elif self.method == 'sliced':
            return self._hellinger_sliced(X_eta, n_projections)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _generate_P_eta_samples(self, eta):
        """
        Generate samples from P_eta = ((1-eta)Id + eta*T)_# P_0
        
        Key insight: We only need to sample X ~ P_0 once, then for any eta:
        Y_eta = (1-eta) * X_test + eta * T(X_test)
        """
        return eta * self.X_test + (1 - eta) * self.X_transported_full
    
    def _hellinger_kde(self, X_eta):
        """KDE-based Hellinger estimation."""
        
        # Fit KDE on P_eta (we pre-fitted P_0)
        kde_eta = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        kde_eta.fit(X_eta)
        
        # Evaluate on union of samples
        X_eval = np.vstack([self.X_train, X_eta])
        
        log_p0 = self.kde_P0.score_samples(X_eval)
        log_p_eta = kde_eta.score_samples(X_eval)
        
        p0 = np.exp(log_p0)
        p_eta = np.exp(log_p_eta)
        
        # Normalize
        p0 = p0 / np.sum(p0)
        p_eta = p_eta / np.sum(p_eta)
        
        # Hellinger distance
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(p0) - np.sqrt(p_eta))**2))
        
        return hellinger
    
    def _hellinger_knn(self, X_eta, k_neighbors):
        """k-NN based Hellinger estimation."""
        
        from sklearn.neighbors import NearestNeighbors
        
        n_eta = len(X_eta)
        X_combined = np.vstack([self.X_train, X_eta])
        
        k = max(3, min(10, int(np.sqrt(self.n_train)))) if k_neighbors is None else k_neighbors
        
        # Fit k-NN
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
        nbrs.fit(X_combined)
        
        distances_train, _ = nbrs.kneighbors(self.X_train)
        distances_eta, _ = nbrs.kneighbors(X_eta)
        
        # Density estimates
        eps = 1e-10
        volume_const = np.pi ** (self.dimension / 2) / np.exp(
            np.sum(np.log(np.arange(1, self.dimension // 2 + 1)))
        )
        
        rho_train_at_train = k / (self.n_train * volume_const * (distances_train[:, k] + eps)**self.dimension)
        rho_eta_at_train = k / (n_eta * volume_const * (distances_train[:, k] + eps)**self.dimension)
        rho_train_at_eta = k / (self.n_train * volume_const * (distances_eta[:, k] + eps)**self.dimension)
        rho_eta_at_eta = k / (n_eta * volume_const * (distances_eta[:, k] + eps)**self.dimension)
        
        # Normalize
        rho_train_at_train = rho_train_at_train / np.sum(rho_train_at_train)
        rho_eta_at_train = rho_eta_at_train / np.sum(rho_eta_at_train)
        rho_train_at_eta = rho_train_at_eta / np.sum(rho_train_at_eta)
        rho_eta_at_eta = rho_eta_at_eta / np.sum(rho_eta_at_eta)
        
        # Hellinger
        term1 = np.sum((np.sqrt(rho_train_at_train) - np.sqrt(rho_eta_at_train))**2)
        term2 = np.sum((np.sqrt(rho_train_at_eta) - np.sqrt(rho_eta_at_eta))**2)
        
        hellinger = np.sqrt(0.25 * (term1 + term2))
        
        return np.clip(hellinger, 0, 1)
    
    def _hellinger_sliced(self, X_eta, n_projections):
        """Sliced Hellinger using random projections."""
        
        np.random.seed(42)
        directions = np.random.randn(n_projections, self.dimension)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        hellinger_distances = []
        
        for direction in directions:
            # Project to 1D
            proj_train = self.X_train @ direction
            proj_eta = X_eta @ direction
            
            # Histogram in 1D
            all_proj = np.concatenate([proj_train, proj_eta])
            n_bins = min(50, max(10, int(np.sqrt(self.n_train))))
            bins = np.linspace(all_proj.min(), all_proj.max(), n_bins + 1)
            
            hist_train, _ = np.histogram(proj_train, bins=bins)
            hist_eta, _ = np.histogram(proj_eta, bins=bins)
            
            # Normalize
            hist_train = hist_train / np.sum(hist_train)
            hist_eta = hist_eta / np.sum(hist_eta)
            
            # Add epsilon
            eps = 1e-10
            hist_train = (hist_train + eps) / np.sum(hist_train + eps)
            hist_eta = (hist_eta + eps) / np.sum(hist_eta + eps)
            
            # 1D Hellinger
            h_1d = np.sqrt(0.5 * np.sum((np.sqrt(hist_train) - np.sqrt(hist_eta))**2))
            hellinger_distances.append(h_1d)
        
        return np.mean(hellinger_distances)
    
    def evaluate_grid(self, eta_values=None, n_projections=50, k_neighbors=5):
        """
        Evaluate Hellinger over a grid of eta values efficiently.
        """
        if eta_values is None:
            eta_values = np.linspace(0, 1, 21)
        
        # print(f"Evaluating Hellinger for {len(eta_values)} eta values...")
        hellinger_distances = self.evaluate(eta_values, n_projections, k_neighbors)
        
        return hellinger_distances
    
    def plot(self, eta_values=None, n_projections=50, k_neighbors=5):
        """
        Plot Hellinger distance vs eta.
        """
        import matplotlib.pyplot as plt
        
        eta_vals, h_dists = self.evaluate_grid(eta_values, n_projections, k_neighbors)
        
        plt.figure(figsize=(10, 6))
        plt.plot(eta_vals, h_dists, 'o-', linewidth=2, markersize=8, color='blue')
        plt.xlabel('η (interpolation parameter)', fontsize=12)
        plt.ylabel('Hellinger Distance H(P₀, Pₙ)', fontsize=12)
        plt.title(f'Hellinger Distance vs η (method={self.method}, d={self.dimension})', 
                 fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return eta_vals, h_dists



def _hellinger_kantorovich(ot_model, eta, X_test, X_train, 
                                   lambda_param=1.0, reg=None,
                                   _global_reg=None, verbose=False):
    """
    HK distance with FIXED regularization across all eta.
    
    Parameters:
    -----------
    reg : float or None
        If None, will be auto-selected ONCE and cached in _global_reg
    _global_reg : dict or None
        Cache for storing the globally selected reg
    """
    
    # Generate samples
    X_trans_full = ot_model.map_treatment_to_control(X_test)
    X_trans = eta * X_test + (1 - eta) * X_trans_full
    
    # Match sample sizes
    n_train = len(X_train)
    n_trans = len(X_trans)
    
    if n_trans != n_train:
        idx = np.random.choice(n_trans, n_train, replace=(n_trans < n_train))
        X_trans = X_trans[idx]
    
    # Mass distributions
    a = np.ones(n_train) / n_train
    b = np.ones(n_train) / n_train
    
    # Cost matrix
    M = cdist(X_train, X_trans, metric='sqeuclidean')
    
    # CRITICAL: Select reg ONCE across all eta values
    if reg is None:
        if _global_reg is None or 'value' not in _global_reg:
            # First call: select reg based on MIDDLE of interpolation (eta=0.5)
            X_mid = 0.5 * X_test + 0.5 * X_trans_full
            if len(X_mid) != n_train:
                idx = np.random.choice(len(X_mid), n_train, replace=(len(X_mid) < n_train))
                X_mid = X_mid[idx]
            M_mid = cdist(X_train, X_mid, metric='sqeuclidean')
            median_dist = np.median(M_mid)
            reg = max(0.1 * median_dist, 0.05)
            
            if _global_reg is not None:
                _global_reg['value'] = reg
            
            if verbose:
                print(f"Selected GLOBAL reg={reg:.4f} (based on eta=0.5, median_dist={median_dist:.4f})")
        else:
            # Subsequent calls: reuse cached reg
            reg = _global_reg['value']
    
    # Mass regularization (fixed by lambda_param)
    lambda_param = np.clip(lambda_param, 0.5, 5.0)
    reg_m = 1.0 / (lambda_param + 1e-6)
    
    try:
        # Solve with FIXED reg
        plan = ot.unbalanced.sinkhorn_knopp_unbalanced(
            a, b, M, reg=reg, reg_m=reg_m,
            numItermax=1000, stopThr=1e-6, verbose=False
        )
        
        if np.isnan(plan).any() or np.isinf(plan).any():
            raise RuntimeError("Plan contains NaN/Inf")
        
        # Compute HK distance
        transport_cost = np.sum(plan * M)
        
        marginal_a = np.sum(plan, axis=1)
        marginal_b = np.sum(plan, axis=0)
        mass_discrepancy = (np.sum(np.abs(a - marginal_a)) + 
                           np.sum(np.abs(b - marginal_b)))
        
        hk_distance = np.sqrt(transport_cost + lambda_param**2 * mass_discrepancy)
        
        if verbose:
            print(f"eta={eta:.2f}: transport={transport_cost:.4f}, "
                  f"mass_disc={mass_discrepancy:.6f}, HK={hk_distance:.4f}")
        
        return hk_distance
        
    except Exception as e:
        warnings.warn(f"HK failed at eta={eta}: {e}")
        return np.nan




def _ipm(ot_model, X_test, X_train, 
                       ipm_metric='w1', kernel='rbf', kernel_bandwidth=1.0):
    """
    Analytical solution for u(eta) = eta with IPM constraint.
    Uses existing packages for distance computation.
    
    Solution: eta* = min(1, c / d(P_X^0, P_X^1))
    
    Parameters:
    -----------
    ot_model : object
        Trained optimal transport model with map_treatment_to_control method
    X_test : array-like, shape (n_test, d)
        Samples from P_X^1 (not used, kept for API consistency)
    X_train : array-like, shape (n_train, d)  
        Samples from P_X^0 (source distribution)
    ipm_metric : str
        'w1' (Wasserstein-1), 'mmd', or 'tv'
    kernel : str
        Kernel for MMD: 'rbf' or 'linear'
    kernel_bandwidth : float
        Bandwidth for RBF kernel
        
    Returns:
    --------
    optimal_eta : float
        Optimal mixing parameter: min(1, c / d(P_X^0, P_X^1))
    optimal_value : float
        Optimal utility value (equals optimal_eta)
    ipm_distance : float
        IPM distance at optimal eta: eta* × d(P_X^0, P_X^1)
    """
    
    n_train = len(X_train)
    dimension = X_train.shape[1]
    
    # Get transport map: T(X_train)
    X_transported = ot_model.map_treatment_to_control(X_train)
    
    # ============================================================
    # Compute d(P_X^0, P_X^1) using existing packages
    # ============================================================
    
    if ipm_metric == 'w1':
        # Wasserstein-1 distance
        
        if dimension == 1:
            # Exact 1D Wasserstein from scipy
            ipm_01 = wasserstein_distance(
                X_train.flatten(), 
                X_transported.flatten()
            )
        elif dimension <= 3:
            # Use POT for exact multi-dimensional Wasserstein
            # Uniform weights
            a = np.ones(n_train) / n_train
            b = np.ones(n_train) / n_train
            
            # Cost matrix (Euclidean distance)
            M = ot.dist(X_train, X_transported, metric='euclidean')
            
            # Compute W1 (EMD)
            ipm_01 = ot.emd2(a, b, M, numItermax=100_000_000)
            
        else:
            ipm_01 = ot.sliced_wasserstein_distance(
                X_train, X_transported, 
                n_projections=100, seed=42,
                numItermax=100_000_000
            )
 
    
    elif ipm_metric == 'mmd':
        # Maximum Mean Discrepancy
        
        if kernel == 'linear':
            # Linear kernel: simple mean difference
            mu_train = np.mean(X_train, axis=0)
            mu_transported = np.mean(X_transported, axis=0)
            ipm_01 = np.linalg.norm(mu_transported - mu_train)
            
        elif kernel == 'rbf':
            # RBF/Gaussian kernel
            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
            
            gamma = 1.0 / (2 * kernel_bandwidth**2)
            
            # Compute kernel matrices using scipy
            K_xx = np.exp(-gamma * cdist(X_train, X_train, 'sqeuclidean'))
            K_yy = np.exp(-gamma * cdist(X_transported, X_transported, 'sqeuclidean'))
            K_xy = np.exp(-gamma * cdist(X_train, X_transported, 'sqeuclidean'))
            
            # MMD^2 unbiased estimator
            n = n_train
            term1 = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1))
            term2 = (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1))
            term3 = np.sum(K_xy) / (n * n)
            
            mmd_squared = term1 + term2 - 2 * term3
            ipm_01 = np.sqrt(np.maximum(mmd_squared, 0))
        
        else:
            raise ValueError(f"Unknown kernel: {kernel}. Use 'linear' or 'rbf'.")
    
    elif ipm_metric == 'tv':
        # Total Variation distance
        # Use k-NN based estimator (works well in high dimensions)
        
        if dimension <= 3:
            # For low dimensions, can use histogram
            n_bins = max(10, int((n_train / 10) ** (1/dimension)))
            n_bins = min(n_bins, 30)
            
            bins_edges = []
            for d in range(dimension):
                min_val = min(X_train[:, d].min(), X_transported[:, d].min())
                max_val = max(X_train[:, d].max(), X_transported[:, d].max())
                bins_edges.append(np.linspace(min_val, max_val, n_bins + 1))
            
            hist_train, _ = np.histogramdd(X_train, bins=bins_edges)
            hist_transported, _ = np.histogramdd(X_transported, bins=bins_edges)
            
            # Normalize
            hist_train = hist_train.flatten() / n_train
            hist_transported = hist_transported.flatten() / n_train
            
            # TV = 0.5 * sum |p - q|
            ipm_01 = 0.5 * np.sum(np.abs(hist_train - hist_transported))
            
        else:
            # k-NN based estimator (better for high-d)
            warnings.warn(
                f"Using k-NN TV estimator for d={dimension}. "
                "This is an approximation."
            )
            
            k = min(5, max(3, n_train // 20))
            
            # Combine both samples
            X_combined = np.vstack([X_train, X_transported])
            labels = np.array([0]*n_train + [1]*n_train)
            
            # Fit k-NN
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
            nbrs.fit(X_combined)
            distances, indices = nbrs.kneighbors(X_combined)
            
            # Count points with same-distribution neighbors
            same_dist_count = 0
            for i in range(len(X_combined)):
                neighbor_labels = labels[indices[i, 1:]]  # Exclude self
                if np.sum(neighbor_labels == labels[i]) > k / 2:
                    same_dist_count += 1
            
            # TV estimate
            ipm_01 = 1.0 - same_dist_count / (2 * len(X_combined))
            ipm_01 = np.clip(ipm_01, 0, 1)
    
    else:
        raise ValueError(
            f"Unknown IPM metric: {ipm_metric}. Use 'w1', 'mmd', or 'tv'."
        )
    
    return ipm_01




class MultidimIPMEstimator:
    def  __init__(self, 
                 max_avg_weight: float = 10.0,
                 n_eta_grid: int = 41,
                 # ot_reg: float = 0.1,
                 ot_estimator=None,
                 # kde_bandwidth: float = None,
                 verbose=False,
                 metric='w2',
                 gaussian=False):
        self.max_avg_weight = max_avg_weight
        self.n_eta_grid = n_eta_grid
        self.ot_model = ot_estimator
        self.metric = metric
        self.gaussian = gaussian
        self.verbose = verbose

        self.eta_star = None
        self.dimension = None
        self.avg_weights = None
        self.ipm_dist = None

    def fit(self, X_test, X_train, **kwargs):
        # if self.gaussian and self.metric in ['w2', 'kl']:
        #     return self._fit_gaussian(X_test, X_train)
        # else:
        return self._fit(X_test, X_train, **kwargs)

    def _fit_gaussian(self, X_test, X_train):
        """Gaussian baseline (same covariates)."""
        mu_test = np.mean(X_test, axis=0)
        mu_train = np.mean(X_train, axis=0)

        n_test, n_train = len(X_test), len(X_train)
        cov_test = np.cov(X_test.T)
        cov_train = np.cov(X_train.T)

        d = X_test.shape[1]
        if d == 1:
            Sigma = (cov_test * n_test + cov_train * n_train) / (n_test + n_train)
            if np.isscalar(Sigma):
                Sigma = np.array([[Sigma]])
        else:
            Sigma = (cov_test * n_test + cov_train * n_train) / (n_test + n_train)
            Sigma += 1e-6 * np.eye(d)

        delta = mu_train - mu_test
        Delta_sq = delta @ np.linalg.solve(Sigma, delta)

        if Delta_sq < 1e-10:
            return 1.0, 1.0

        if self.metric == 'w2':
            eta = np.sqrt(self.max_avg_weight / Delta_sq)
        elif self.metric == 'kl':
            eta = np.sqrt(2*self.max_avg_weight / Delta_sq)
        print("gaussian eta", eta)
        eta = np.clip(eta, 0, 1)
        w = self.max_avg_weight

        if self.verbose:
            print(f"  Δ² = {Delta_sq:.4f}, η* = {eta:.4f}, E[w] = {w:.2f}")
        
        self.eta_star = eta
        self.avg_weights = w
        return self


    def _fit(self, X_test: np.ndarray, X_train: np.ndarray, **kwargs):
        X_test = np.atleast_2d(X_test)
        X_train = np.atleast_2d(X_train)
        
        self.dimension = X_test.shape[1]
    
        # Grid search
        self.eta_grid = np.linspace(0, 1, self.n_eta_grid)
        # self.eta_grid = eta_grid_dense_at_zero(self.n_eta_grid, power=2)
        self.avg_weights = []
        

        if self.metric == "hellinger":

            estimator = HellingerEstimator(self.ot_model, X_test, X_train, **kwargs)
            self.avg_weights = estimator.evaluate_grid(self.eta_grid)
            self.max_avg_weight = np.quantile(self.avg_weights, q=q)
            
        else:
            X_trans_full = self.ot_model.map_treatment_to_control(X_test)
            for i, eta in enumerate(self.eta_grid):
                X_trans = eta * X_test + (1 - eta) * X_trans_full
                
                # Transport test samples
                if self.metric == "w1":
                    
                    avg_weight = _w1(X_train, X_trans)

                elif self.metric == "w2":
                   
                    avg_weight = _w2(X_train, X_trans)

                elif self.metric == "mmd":
                    avg_weight = _mmd(X_train, X_trans, gamma=1.)

                self.avg_weights.append(avg_weight)
            
            self.avg_weights = np.array(self.avg_weights)        
        # print(" average weights", self.avg_weights)

        self.ipm_dist = self.avg_weights[-1]
        # Find optimal η
        feasible = self.avg_weights <= self.max_avg_weight
        if np.any(feasible):
            self.eta_star = self.eta_grid[feasible].max()
        else:
            warnings.warn(f"No feasible η, min weight: {self.avg_weights.min():.2f}")
            self.eta_star = self.eta_grid[np.argmin(self.avg_weights)]
        
        idx_star = np.argmin(np.abs(self.eta_grid - self.eta_star))
        avg_weights_star = self.avg_weights[idx_star]
        
        if self.verbose:
            print(self.avg_weights)
            print(f"  Optimal η* = {self.eta_star:.4f}")
            print(f"  E[w_η*] = {avg_weights_star:.2f}")
            print(f"  0 to 1 dist = {self.ipm_dist: 4f}")
        
        return self

    def transform(self, X_test: np.ndarray, eta: Optional[float] = None):
        """Apply geodesic transport."""
        if eta is None:
            eta = self.eta_star
        
        X_trans_full = self.ot_model.map_treatment_to_control(X_test)
        return eta * X_test + (1 - eta) * X_trans_full

# class DualDensityRatioEstimator:
#     """
#     Dual formulation for density ratio estimation using classification.
    
#     Key idea: To estimate w(y) = p_train(y) / p_η(y), train a classifier
#     to distinguish samples from P_train vs P_η.
    
#     For binary classification with equal priors:
#         p(y|class=train) / p(y|class=η) = P(class=train|y) / P(class=η|y)
    
#     Therefore: w(y) ≈ h(y) / (1 - h(y))
#     where h(y) = P(class=train|y) from the classifier.
    
#     This avoids curse of dimensionality by using discriminative models!
#     """
    
#     def __init__(self,
#                  max_avg_weight: float = 10.0,
#                  n_eta_grid: int = 30,
#                  ot_model=None,
#                  # ot_reg: float = 0.1,
#                  # method: str = "emd",
#                  classifier_type: str = 'logistic',
#                  alpha: float = 0.01, gaussian=False,
#                  verbose=False):
#         """
#         Parameters:
#         -----------
#         classifier_type : str
#             'logistic': Logistic regression (linear, fast, works for d~100)
#             'mlp': Neural network (nonlinear, works for d~1000)
#         alpha : float
#             Smoothing parameter: w(y) = h(y)/(1-h(y)+α) to avoid division by 0
#         """
#         self.max_avg_weight = max_avg_weight
#         self.n_eta_grid = n_eta_grid
#         # self.ot_reg = ot_reg
#         # self.method = method
#         self.ot_model = ot_model
#         self.classifier_type = classifier_type
#         self.alpha = alpha
#         self.gaussian = gaussian
#         self.verbose=verbose
        
#         self.eta_star = None
#         self.dimension = None
#         self.eta_grid = None
#         self.avg_weights = None

        
#     def fit(self, X_test: np.ndarray, X_train: np.ndarray):
#         """Fit using dual density ratio estimation."""
#         X_test = np.atleast_2d(X_test)
#         X_train = np.atleast_2d(X_train)
        
#         self.dimension = X_test.shape[1]
        
#         # # fit OT map
#         # if self.method == 'sinkhorn':
#         #     self.ot_model = ot.da.SinkhornTransport(reg_e=self.ot_reg)
#         # elif self.method == "emd":
#         #     self.ot_model = ot.da.EMDTransport(max_iter=100_000_000)
#         # else:
#         #     raise ValueError("Choose method between 'sinkhorn' or 'emd'.")
#         # self.ot_model.fit(Xs=X_test, Xt=X_train)


#         if self.gaussian:
#             # closed form
#             self.eta_star, self.avg_weights = self._fit_gaussian(X_test, X_train)
#         else:
#             # print(f"Dual Formulation (d={self.dimension}):")
#             # print(f"  n_test={len(X_test)}, n_train={len(X_train)}")
#             # print(f"  Classifier: {self.classifier_type}")
            
#             # Grid search
#             self.eta_grid = np.linspace(0, 1, self.n_eta_grid)
#             self.avg_weights = []
            
#             for i, eta in enumerate(self.eta_grid):
#                 avg_weight = self._estimate_avg_weight_classification(eta, X_test, X_train)
#                 self.avg_weights.append(avg_weight)
                
#                 if i % 10 == 0:
#                     print(f"    η={eta:.2f}: E[w]={avg_weight:.2f}")
            
#             self.avg_weights = np.array(self.avg_weights)
            
#             # Find optimal η
#             feasible = self.avg_weights <= self.max_avg_weight
#             if np.any(feasible):
#                 self.eta_star = self.eta_grid[feasible].max()
#             else:
#                 print(f"    WARNING: No feasible η (min E[w]={self.avg_weights.min():.2f})")
#                 self.eta_star = self.eta_grid[np.argmin(self.avg_weights)]
            
#             idx_star = np.argmin(np.abs(self.eta_grid - self.eta_star))
#             if self.verbose:
#                 print(f"  ✓ Optimal η* = {self.eta_star:.4f}")
#                 print(f"  ✓ E[w_η*] = {self.avg_weights[idx_star]:.2f}")
            
#         return self
    
    







class ComprehensiveComparison:
    """Compare all methods: Gaussian, Plug-in KDE, Dual."""
    
    def __init__(self, max_avg_weight: float = 10.0):
        self.max_avg_weight = max_avg_weight
        self.results = {}
        
    def fit_all(self, X_test: np.ndarray, X_train: np.ndarray):
        """Fit all three methods."""
        X_test = np.atleast_2d(X_test)
        X_train = np.atleast_2d(X_train)

        # Fit OT map
        ot_model = ot.da.EMDTransport()
        ot_model.fit(Xs=X_test, Xt=X_train)
        
        d = X_test.shape[1]
        
        print("="*70)
        print(f"COMPREHENSIVE COMPARISON (d={d})")
        print("="*70)
        
        # Method 1: Gaussian
        print("\n[1/4] Gaussian closed-form...")
        # eta_gauss, w_gauss = self._fit_gaussian(X_test, X_train)
        dual = MultidimRatioEstimator(
            max_avg_weight=self.max_avg_weight, ot_estimator=ot_model,
            method='gaussian', verbose=True
        )
        dual.fit(X_test, X_train)
        self.results['gaussian'] = {
        'eta': dual.eta_star, 
        'avg_weight': dual.avg_weights,
        'estimator': dual}
        
        # Method 2: Plug-in KDE (if low-d)
        if d <= 4:
            print("\n[2/4] Plug-in KDE...")
            try:
                plugin = MultidimRatioEstimator(
                    max_avg_weight=self.max_avg_weight, ot_estimator=ot_model,
                    method='kde', verbose=True
                )
                plugin.fit(X_test, X_train)
                self.results['plugin'] = {
                    'eta': plugin.eta_star,
                    'avg_weight': plugin.avg_weights[np.argmin(np.abs(plugin.eta_grid - plugin.eta_star))],
                    'estimator': plugin
                }
            except Exception as e:
                print(f"  Plug-in KDE failed: {e}")
                self.results['plugin'] = None
        else:
            print("\n[2/4] Plug-in KDE skipped (d>4)")
            self.results['plugin'] = None
        
        # Method 3: Dual
        print("\n[3/4] Dual classifier formulation...")
        classifier = MultidimRatioEstimator(
            max_avg_weight=self.max_avg_weight, ot_estimator=ot_model,
            method='classifier', verbose=True
        )
        classifier.fit(X_test, X_train)
        self.results['classifier'] = {
            'eta': classifier.eta_star,
            'avg_weight': classifier.avg_weights[np.argmin(np.abs(classifier.eta_grid - classifier.eta_star))],
            'estimator': classifier
        }

        # Method 4: ULSIF
        print("\n[4/4] Dual ULSIF formulation...")
        dual = MultidimRatioEstimator(
            max_avg_weight=self.max_avg_weight, ot_estimator=ot_model,
            method='ulsif', verbose=True
        )
        dual.fit(X_test, X_train)
        self.results['ulsif'] = {
            'eta': dual.eta_star,
            'avg_weight': dual.avg_weights[np.argmin(np.abs(dual.eta_grid - dual.eta_star))],
            'estimator': dual
        }
        
        return self
    
    def print_summary(self):
        """Print comparison table."""
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"{'Method':<20} | {'η*':>8} | {'E[w_η*]':>10} | {'Status':>15}")
        print("-"*70)
        
        for name, key in [('Gaussian', 'gaussian'), 
                          ('Plug-in KDE', 'plugin'),
                          ('Logistic', 'classifier'),
                          ('ULSIF', 'ulsif')]:
            if self.results[key] is not None:
                r = self.results[key]
                print(f"{name:<20} | {r['eta']:8.4f} | {r['avg_weight']:10.2f} | {'✓':>15}")
            else:
                print(f"{name:<20} | {'N/A':>8} | {'N/A':>10} | {'Skipped':>15}")
        
        print("="*70)


# Examples
if __name__ == "__main__":
    np.random.seed(42)
    
    # Example 1: 2D (all methods applicable)
    print("\n" + "="*70)
    print("EXAMPLE 1: 2D Gaussian")
    print("="*70)
    
    mu_test = np.array([-2, -3])
    mu_train = np.array([1, 1])
    Sigma = np.array([[1, 0.3], [0.3, 1]])
    
    n = 6000
    X_test_2d = np.random.multivariate_normal(mu_test, Sigma, n)
    X_train_2d = np.random.multivariate_normal(mu_train, Sigma, n)
    
    comp_2d = ComprehensiveComparison(max_avg_weight=50.0)
    comp_2d.fit_all(X_test_2d, X_train_2d)
    comp_2d.print_summary()
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    for i, (name, key) in enumerate([('Gaussian', 'gaussian'),
                                       ('Plug-in KDE', 'plugin'),
                                       ('Logistic', 'classifier'),
                                       ('ULSIF', 'ulsif')]):
        ax = axes[i]
        
        if comp_2d.results[key] is not None and 'estimator' in comp_2d.results[key]:
            est = comp_2d.results[key]['estimator']
            X_trans = est.transform(X_test_2d)
            
            ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], alpha=0.3, s=10, 
                      c='red', label='Test')
            ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], alpha=0.3, s=10,
                      c='blue', label='Train')
            ax.scatter(X_trans[:, 0], X_trans[:, 1], alpha=0.3, s=10,
                      c='green', label='Transported')
            
            ax.set_title(f'{name}\nη*={comp_2d.results[key]["eta"]:.3f}',
                        fontweight='bold')
        # else:
        #     ax.text(0.5, 0.5, f'{name}\n(Gaussian only)',
        #            ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('comparison_2d_all_methods.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: comparison_2d_all_methods.png")
    
    # Example 2: 10D (only Gaussian + Dual work well)
    print("\n" + "="*70)
    print("EXAMPLE 2: 10D Gaussian (High-dimensional)")
    print("="*70)
    
    # d = 10
    # mu_test_10d = np.zeros(d)
    # mu_train_10d = np.ones(d) * 1.5
    # Sigma_10d = np.eye(d) + 0.1 * np.ones((d, d))
    
    # n = 800
    # X_test_10d = np.random.multivariate_normal(mu_test_10d, Sigma_10d, n)
    # X_train_10d = np.random.multivariate_normal(mu_train_10d, Sigma_10d, n)
    
    # comp_10d = ComprehensiveComparison(max_avg_weight=10.0)
    # comp_10d.fit_all(X_test_10d, X_train_10d)
    # comp_10d.print_summary()
    
    # # Dimensional scaling study
    # print("\n" + "="*70)
    # print("DIMENSIONAL SCALING STUDY")
    # print("="*70)
    
    # dimensions = [2, 3, 5]
    # results_by_dim = {
    #     'gaussian': [],
    #     'dual': []
    # }
    
    # for d in dimensions:
    #     print(f"\nDimension d={d}...")
        
    #     # Fixed Mahalanobis distance per dimension
    #     mu_test_d = np.zeros(d)
    #     mu_train_d = np.ones(d) * 1.5
    #     Sigma_d = np.eye(d)
        
    #     X_test_d = np.random.multivariate_normal(mu_test_d, Sigma_d, 600)
    #     X_train_d = np.random.multivariate_normal(mu_train_d, Sigma_d, 600)
        
    #     comp_d = ComprehensiveComparison(max_avg_weight=10.0)
    #     comp_d.fit_all(X_test_d, X_train_d)
        
    #     results_by_dim['gaussian'].append(comp_d.results['gaussian']['eta'])
    #     results_by_dim['dual'].append(comp_d.results['dual']['eta'])
    
    # # Plot dimensional scaling
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # # Plot 1: η vs dimension
    # ax = axes[0]
    # ax.plot(dimensions, results_by_dim['gaussian'], 'ro-', 
    #        linewidth=2.5, markersize=10, label='Gaussian (theory)')
    # ax.plot(dimensions, results_by_dim['dual'], 'bs-',
    #        linewidth=2.5, markersize=10, label='Dual (empirical)')
    # ax.set_xlabel('Dimension d', fontsize=13)
    # ax.set_ylabel('Optimal η*', fontsize=13)
    # ax.set_title('Scalability: η* vs Dimension', fontsize=14, fontweight='bold')
    # ax.legend(fontsize=11)
    # ax.grid(True, alpha=0.3)
    
    # # Plot 2: Agreement between methods
    # ax = axes[1]
    # agreement = [abs(g - d) for g, d in zip(results_by_dim['gaussian'], 
    #                                           results_by_dim['dual'])]
    # ax.plot(dimensions, agreement, 'go-', linewidth=2.5, markersize=10)
    # ax.set_xlabel('Dimension d', fontsize=13)
    # ax.set_ylabel('|η*_Gaussian - η*_Dual|', fontsize=13)
    # ax.set_title('Method Agreement', fontsize=14, fontweight='bold')
    # ax.grid(True, alpha=0.3)
    # ax.set_yscale('log')
    
    # plt.tight_layout()
    # plt.savefig('dimensional_scaling.png', dpi=150, bbox_inches='tight')
    # print("\nPlot saved: dimensional_scaling.png")
    
    # # Summary table
    # print("\n" + "="*70)
    # print("DIMENSIONAL SCALING TABLE")
    # print("="*70)
    # print(f"{'Dim':>5} | {'η* (Gauss)':>12} | {'η* (Dual)':>12} | {'Difference':>12}")
    # print("-"*70)
    # for d, eta_g, eta_d in zip(dimensions, results_by_dim['gaussian'], 
    #                             results_by_dim['dual']):
    #     diff = abs(eta_g - eta_d)
    #     print(f"{d:5d} | {eta_g:12.4f} | {eta_d:12.4f} | {diff:12.4f}")
    # print("="*70)
    
    # print("\n" + "="*70)
    # print("KEY FINDINGS:")
    # print("="*70)
    # print("1. Plug-in KDE works well for d ≤ 3-4")
    # print("2. Dual formulation scales to high dimensions (tested up to d=50)")
    # print("3. Dual matches Gaussian theory when data is Gaussian")
    # print("4. For non-Gaussian data, Dual is more accurate than Gaussian assumption")
    # print("5. Computational cost: Gaussian < Dual < Plug-in KDE")
    # print("="*70)