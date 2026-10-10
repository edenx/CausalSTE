import numpy as np
from typing import Tuple, Optional, Union

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
except Exception:
    Pipeline = object  # type: ignore
    PolynomialFeatures = object  # type: ignore
    StandardScaler = object  # type: ignore

RegressorLike = Union[object]

def gradient_eval(func_type: str,
                  X: np.ndarray,
                  coef: np.ndarray,
                  bias: float = 0.0,
                  A: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized gradient + value for linear/quadratic models.

    Parameters
    ----------
    func_type : {'linear', 'quadratic'}
    X : (n, d) or (d,) array
        Points where to evaluate.
    coef : (d,) array
        Linear coefficient (beta).
    bias : float
        Intercept term (c).
    A : (d, d) array, optional
        Symmetric quadratic matrix (required for 'quadratic').

    Returns
    -------
    grads : (n, d) array
        Row i is ∇f(X[i]). For linear, rows are copies of `coef`.
    values : (n,) array
        f(X[i]) values.

    Notes
    -----
    We assume the canonical form:
      Linear:    f(x) = beta^T x + c
      Quadratic: f(x) = 0.5 x^T A x + beta^T x + c
    """
    X = np.asarray(X)
    coef = np.asarray(coef)

    # Normalize to (n, d)
    squeeze_back = False
    if X.ndim == 1:
        X = X[None, :]
        squeeze_back = True

    if func_type == 'linear':
        values = X @ coef + bias  # (n,)
        grads = coef[:, None]
        return grads, values

    if func_type == 'quadratic':
        if A is None:
            raise ValueError("A must be provided for quadratic")
        A = np.asarray(A)
        # Values: 0.5 x^T A x + beta^T x + c
        values = 0.5 * np.einsum('ni,ij,nj->n', X, A, X) + X @ coef + bias
        # Gradients: A x + beta   → vectorized as X @ A^T + beta
        grads = X @ A.T
        grads += coef  # broadcasts (d,) across rows
        return grads, values

    raise ValueError("func_type must be 'linear' or 'quadratic'")

# -----------------------------
# Linear models: y = beta^T x + c
# -----------------------------

def extract_linear_params(model: RegressorLike,
                          scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, float]:
    """
    Extract (beta, c) from a fitted *linear* scikit-learn model.

    Supports any regressor exposing `.coef_` and `.intercept_` with shape (d,) and float.

    If a StandardScaler (fitted) was applied BEFORE the linear model, pass it via
    `scaler` to obtain parameters in the ORIGINAL x-space.

    Given z = (x - m) / s, and model y = w^T z + b, the original-space parameters are:
        beta = w / s
        c    = b - (w / s)^T m
    where division is elementwise.
    """
    w = np.asarray(getattr(model, "coef_"))
    if w.ndim > 1:
        raise ValueError("Only single-output regressors supported (coef_.ndim must be 1).")
    b = float(getattr(model, "intercept_", 0.0))

    if scaler is None:
        return w.copy(), b

    if not hasattr(scaler, "scale_") or not hasattr(scaler, "mean_"):
        raise ValueError("Provided scaler must be a fitted sklearn.preprocessing.StandardScaler.")

    s = np.asarray(scaler.scale_)
    m = np.asarray(scaler.mean_)
    beta = w / s
    c = b - float(beta @ m)
    return beta, c


# ----------------------------------------------------------
# Quadratic models via PolynomialFeatures(degree=2)
# y = 0.5 x^T A x + beta^T x + c
# ----------------------------------------------------------

def _reconstruct_quadratic_from_poly(linear_model: RegressorLike,
                                     poly: PolynomialFeatures,
                                     feature_names: Optional[Union[list, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Reconstruct (A, beta, c) from a LinearRegression-like model fitted on
    PolynomialFeatures(degree=2) of the ORIGINAL features (no scaling inside here).

    Mapping assumes scikit-learn's feature name conventions:
      - linear term  :  x_i         -> contributes to beta[i]
      - square term  :  x_i^2       -> contributes to A[i,i]  via  A[i,i] = 2 * coef
      - interaction  :  x_i x_j     -> contributes to A[i,j] and A[j,i] with value `coef`
      - constant/bias:  1           -> contributes directly to c

    Returns
    -------
    A : (d, d) symmetric ndarray
    beta : (d,) ndarray
    c : float
    """
    if int(getattr(poly, "degree", 0)) != 2:
        raise ValueError("PolynomialFeatures must have degree=2.")
    if getattr(poly, "interaction_only", False):
        raise ValueError("interaction_only=True omits squares; set interaction_only=False for full quadratic.")

    # Determine original dimension and feature names
    d = int(getattr(poly, "n_features_in_", 0) or getattr(poly, "n_input_features_", 0))
    if d <= 0:
        raise ValueError("PolynomialFeatures appears unfitted; missing n_features_in_. Fit it first.")

    if feature_names is None:
        try:
            feature_names = list(poly.feature_names_in_)
        except Exception:
            feature_names = [f"x{i}" for i in range(d)]

    # Names for the expanded polynomial features in the *same* order as coef_
    names_out = list(poly.get_feature_names_out(feature_names))

    # Fetch linear model params on poly features
    coefs = np.asarray(getattr(linear_model, "coef_"))
    if coefs.ndim > 1:
        raise ValueError("Only single-output regressors supported (coef_.ndim must be 1).")
    intercept = float(getattr(linear_model, "intercept_", 0.0))

    A = np.zeros((d, d), dtype=float)
    beta = np.zeros(d, dtype=float)
    c = float(intercept)

    # Helper: map feature name to (i,j) or i
    # Expected patterns from scikit-learn when feature_names=['x0','x1',...] or real names
    for name, coef in zip(names_out, coefs):
        if coef == 0.0:
            continue
        if name == "1":  # bias term from poly(include_bias=True)
            c += float(coef)
            continue
        # Linear: exact match to one original feature name
        if name in feature_names:
            i = feature_names.index(name)
            beta[i] += float(coef)
            continue
        # Squares: like 'x0^2' or 'feat^2'
        if name.endswith("^2"):
            base = name[:-2]
            if base in feature_names:
                i = feature_names.index(base)
                A[i, i] += 2.0 * float(coef)
                continue
        # Interactions: 'x0 x1' or 'featA featB' (separated by space)
        if " " in name:
            a, b = name.split(" ")
            if a in feature_names and b in feature_names:
                i = feature_names.index(a)
                j = feature_names.index(b)
                # symmetric placement so that 0.5 * (A_ij + A_ji) * x_i x_j = coef * x_i x_j
                A[i, j] += float(coef)
                A[j, i] += float(coef)
                continue
        # If we reach here, name pattern wasn't recognized; ignore silently

    # Enforce symmetry numerically (average with transpose)
    A = 0.5 * (A + A.T)
    return A, beta, c


def extract_quadratic_params(
    model_or_pipeline: Union[RegressorLike, Pipeline],
    poly: Optional[PolynomialFeatures] = None,
    scaler: Optional[StandardScaler] = None,
    feature_names: Optional[Union[list, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract (A, beta, c) for a quadratic predictor in the ORIGINAL x-space.

    Two supported setups:
    1) Separate objects: poly = PolynomialFeatures(degree=2), model = LinearRegression(...)
       (optionally, a `scaler` that was applied BEFORE `poly`).
    2) A fitted Pipeline like: Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(2)), ('lin', LinearRegression())])

    If a StandardScaler was applied BEFORE PolynomialFeatures, we convert parameters from z-space to x-space via:
        A = S^{-T} A' S^{-1}
        beta = S^{-T} beta' - A m
        c = c' - (beta')^T S^{-1} m + 0.5 m^T A m
    where z = (x - m) / s, S = diag(s), and (A', beta', c') are parameters in z-space.
    """
    pipe = None
    lin = None
    z_poly = None
    z_scaler = None

    if isinstance(model_or_pipeline, Pipeline):
        pipe = model_or_pipeline
        # Find steps
        for name, step in pipe.steps:
            if isinstance(step, PolynomialFeatures):
                z_poly = step
            elif isinstance(step, StandardScaler):
                # We assume scaler is BEFORE poly in the pipeline
                z_scaler = step
            else:
                # last step expected to be linear regressor
                lin = step
        if lin is None or z_poly is None:
            raise ValueError("Pipeline must contain PolynomialFeatures and a linear regressor as final step.")
    else:
        lin = model_or_pipeline
        z_poly = poly
        z_scaler = scaler
        if z_poly is None:
            raise ValueError("Provide a fitted PolynomialFeatures(degree=2) transformer via `poly`.")

    # First reconstruct in the feature space that PolynomialFeatures saw (z-space if scaler was applied)
    A_prime, beta_prime, c_prime = _reconstruct_quadratic_from_poly(lin, z_poly, feature_names=feature_names)

    # If no scaler before poly, these are already original-space params
    if z_scaler is None:
        return A_prime, beta_prime, c_prime

    # Convert from z-space to x-space
    s = np.asarray(z_scaler.scale_)
    m = np.asarray(z_scaler.mean_)
    S_inv = np.diag(1.0 / s)

    # A = S^{-T} A' S^{-1}
    A = S_inv.T @ A_prime @ S_inv
    # beta = S^{-T} beta' - A m
    beta = S_inv.T @ beta_prime - A @ m
    # c = c' - (beta')^T S^{-1} m + 0.5 m^T A m
    c = float(c_prime - beta_prime @ (S_inv @ m) + 0.5 * (m @ (A @ m)))
    return A, beta, c


# -----------------------------
# Convenience wrappers & checks
# -----------------------------

def check_quadratic_grad(A: np.ndarray, beta: np.ndarray, c: float, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Given (A, beta, c) and a point x, compute gradient and value:
        f(x) = 0.5 x^T A x + beta^T x + c
        grad = A x + beta
    Useful to plug directly into your gradient_eval implementation.
    """
    x = np.asarray(x)
    grad = A @ x + beta
    val = 0.5 * float(x @ (A @ x)) + float(beta @ x) + float(c)
    return grad, val


