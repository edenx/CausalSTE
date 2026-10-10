import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, Tuple
from .ot_maps import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class ProbabilityWrapper:
    """
    Wrapper for classifiers to ensure .predict() returns probabilities P(Y=1)
    instead of class labels. Handles edge case where only one class is present.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        # Extract classes from already-fitted classifier
        self.classes_seen_ = getattr(classifier, 'classes_', None)
    
    def fit(self, X, y):
        self.classifier.fit(X, y)
        self.classes_seen_ = self.classifier.classes_
        return self
    
    def predict(self, X):
        """Return probability of positive class P(Y=1)"""
        proba = self.classifier.predict_proba(X)
        
        # Handle edge case: only one class seen during training
        if proba.shape[1] == 1:
            # Check which class was seen
            if self.classes_seen_[0] == 1:
                # Only saw class 1, so P(Y=1) = 1.0
                return proba[:, 0]
            else:
                # Only saw class 0, so P(Y=1) = 0.0
                return 1 - proba[:, 0]
        else:
            # Normal case: both classes seen, return P(Y=1)
            return proba[:, 1]
    
    def predict_proba(self, X):
        """
        Return full probability matrix, ensuring it's always (n_samples, 2)
        """
        proba = self.classifier.predict_proba(X)
        
        # Handle edge case: only one class seen during training
        if proba.shape[1] == 1:
            if self.classes_seen_[0] == 1:
                # Only saw class 1: P(Y=0)=0, P(Y=1)=1
                return np.column_stack([1 - proba[:, 0], proba[:, 0]])
            else:
                # Only saw class 0: P(Y=0)=1, P(Y=1)=0
                return np.column_stack([proba[:, 0], 1 - proba[:, 0]])
        else:
            return proba

def create_stratified_splits(X, Y, K, random_state=42):
    """
    Create stratified K-fold splits for binary outcomes.
    Ensures both classes appear in each fold.
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    splits = []
    
    for train_idx, test_idx in skf.split(X, Y):
        splits.append((train_idx, test_idx))
    
    return splits

class LogLinkWrapper:
    """
    Wrapper that models log(Y) but returns predictions on the original scale.
    Ensures positive predictions for survival time.
    """
    def __init__(self, regressor):
        self.regressor = regressor
    
    def fit(self, X, y):
        """
        Fit on log-transformed outcome.
        y should already be log-transformed before passing to fit.
        """
        self.regressor.fit(X, y)
        return self
    
    def predict(self, X):
        """Return predictions on original scale: exp(prediction)"""
        log_pred = self.regressor.predict(X)
        return np.exp(log_pred)

class CrossFitData:
    """
    Container for cross-fitting data splits and transport maps.
    Ensures consistent splits across different estimation methods.
    """
    def __init__(self, X_control, X_treatment, Y_control, Y_treatment, 
                 Y0_treatment, true_ATT, K=5, random_state=42, 
                 binary_outcome=False, use_log_link=False):
        """
        Initialize cross-fitting data structure.
        
        Parameters:
        -----------
        X_control, X_treatment : arrays
            Covariate data
        Y_control, Y_treatment : arrays
            Outcome data
        Y0_treatment : array
            True counterfactual outcomes (for evaluation)
        true_ATT : float
            True average treatment effect on treated
        K : int
            Number of folds
        random_state : int
            Random seed for reproducibility
        """
        self.X_control = X_control
        self.X_treatment = X_treatment
        self.Y_control = Y_control
        self.Y_treatment = Y_treatment
        self.Y0_treatment = Y0_treatment
        self.true_ATT = true_ATT
        self.K = K
        self.random_state = random_state
        self.binary_outcome = binary_outcome
        self.use_log_link = use_log_link
        
        if binary_outcome:
            # Use stratified splits for binary outcomes
            self.control_splits = create_stratified_splits(
                self.X_control, self.Y_control, K, random_state
            )
            self.treatment_splits = create_stratified_splits(
                self.X_treatment, self.Y_treatment, K, random_state
            )
        else:
            # Create fold indices
            self.kf_control = KFold(n_splits=K, shuffle=True, random_state=random_state)
            self.kf_treatment = KFold(n_splits=K, shuffle=True, random_state=random_state)
            
            # Store fold splits
            self.control_splits = list(self.kf_control.split(X_control))
            self.treatment_splits = list(self.kf_treatment.split(X_treatment))
                
        # Transport estimators (one per fold)
        self.transport_estimators = [None] * K
        self.outcome_estimators = [None] * K
        
    def fit_transport_maps(self, verbose=False, **ot_params):
        """
        Fit transport maps for all folds.
        
        Parameters:
        -----------
        ot_class : class
            OptimalTransportEstimator class
        ot_params : dict
            Parameters to pass to ot_class constructor
        verbose : bool
            Print progress
        """
        if verbose:
            print(f"\nFitting transport maps for {self.K} folds...")
            print(f"OT params: {ot_params}")
        
        for fold_idx in range(self.K):
            if verbose:
                print(f"\nFold {fold_idx + 1}/{self.K}")
            
            # Get training indices for this fold
            train_ctrl_idx, _ = self.control_splits[fold_idx]
            train_trt_idx, _ = self.treatment_splits[fold_idx]
            
            # Create and fit transport estimator
            ot_estimator = OptimalTransportEstimator(**ot_params)
            ot_estimator.estimate_inverse_transport(
                self.X_control[train_ctrl_idx],
                self.X_treatment[train_trt_idx],
                verbose=verbose
            )
            
            self.transport_estimators[fold_idx] = ot_estimator
        
        if verbose:
            print(f"\n✓ All {self.K} transport maps fitted")



    def fit_outcome_models(self, outcome_model_type, degree=3, 
                          max_depth=None, min_samples_leaf=5,
                          n_estimators=500, random_state=42
                          ):  # NEW
        """
        Fit outcome model on control data
        
        Parameters:
        -----------
        binary_outcome : bool
            If True, use classification models
        use_log_link : bool
            If True, model log(Y) for positive continuous outcomes.
            Only applies to linear/polynomial models.
            Predictions will be exponentiated to ensure positivity.
        """
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        for fold_idx in range(self.K):
            train_ctrl_idx, _ = self.control_splits[fold_idx]
            
            X_train = self.X_control[train_ctrl_idx]
            y_train = self.Y_control[train_ctrl_idx]
            
            # Transform outcome for log-link
            if self.use_log_link and not self.binary_outcome:
                # Add small constant to avoid log(0)
                y_train_transformed = np.log(y_train + 1e-6)
            else:
                y_train_transformed = y_train
            
            # Check for single-class issue in binary outcomes
            if self.binary_outcome:
                unique_classes = np.unique(y_train)
                if len(unique_classes) < 2:
                    print(f"Warning: Fold {fold_idx} has only one class. Using constant model.")
                    outcome_model = ConstantClassifier(constant_value=float(unique_classes[0]))
                    outcome_model.fit(X_train, y_train)
                    self.outcome_estimators[fold_idx] = outcome_model
                    continue
            
            # Build model
            if outcome_model_type == 'linear':
                if self.binary_outcome:
                    outcome_model = LogisticRegression(max_iter=1e10, random_state=random_state)
                else:
                    outcome_model = LinearRegression()
                
            elif outcome_model_type == 'quadratic':
                if self.binary_outcome:
                    outcome_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                        ('logistic', LogisticRegression(max_iter=1e10, random_state=random_state))
                    ])
                else:
                    outcome_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
                
            elif outcome_model_type == 'polynomial':
                if self.binary_outcome:
                    outcome_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                        ('logistic', LogisticRegression(max_iter=1e10, random_state=random_state))
                    ])
                else:
                    outcome_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
            elif outcome_model_type == "xgboost":
                # XGBoost (very popular)
                outcome_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01)

            elif outcome_model_type == 'ridge':

                outcome_model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                    ('ridge', Ridge(alpha=5.0))  # alpha controls regularization strength
                ])
            elif outcome_model_type == "elastic":
                outcome_model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                    ('elastic', ElasticNet(alpha=1.0, l1_ratio=0.5))
                ])
            
            elif outcome_model_type == 'tree':
                if self.binary_outcome:
                    outcome_model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state
                    )
                else:
                    outcome_model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state
                    )
                
            elif outcome_model_type == 'forest':
                if self.binary_outcome:
                    outcome_model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        n_jobs=-1
                    )
                else:
                    outcome_model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        n_jobs=-1
                    )
            else:
                raise ValueError(f"Unknown outcome_model_type: {outcome_model_type}")
            
            # Wrap classifiers
            if self.binary_outcome and outcome_model_type in ['tree', 'forest']:
                outcome_model = ProbabilityWrapper(outcome_model)
            
            # Wrap with log-link for positive continuous outcomes (linear models only)
            if self.use_log_link and not self.binary_outcome and outcome_model_type in ['linear', 'quadratic', 'polynomial']:
                outcome_model = LogLinkWrapper(outcome_model)
            
            # Fit
            try:
                outcome_model.fit(X_train, y_train_transformed)
            except ValueError as e:
                if self.binary_outcome and "only one class" in str(e):
                    print(f"Warning: Fold {fold_idx} failed. Using constant model.")
                    outcome_model = ConstantClassifier(constant_value=y_train.mean())
                    outcome_model.fit(X_train, y_train)
                else:
                    raise e
            
            self.outcome_estimators[fold_idx] = outcome_model



    def get_fold_data(self, fold_idx):
        """
        Get train/test data for a specific fold.
        
        Returns:
        --------
        dict with keys:
            - X_ctrl_train, Y_ctrl_train
            - X_trt_train, Y_trt_train  
            - X_trt_test, Y_trt_test, Y0_trt_test
            - train_ctrl_idx, test_ctrl_idx
            - train_trt_idx, test_trt_idx
            - transport_estimator
        """
        train_ctrl_idx, test_ctrl_idx = self.control_splits[fold_idx]
        train_trt_idx, test_trt_idx = self.treatment_splits[fold_idx]
        
        transport_estimator = self.transport_estimators[fold_idx]
        X_trt_test = self.X_treatment[test_trt_idx]
        X_trt_train = self.X_treatment[train_trt_idx]
        # Map test treatment samples using pre-fitted transport
        X0_test = transport_estimator.map_treatment_to_control(X_trt_test)
        X0_train = transport_estimator.map_treatment_to_control(X_trt_train)

        return {
            'X_ctrl_train': self.X_control[train_ctrl_idx],
            'Y_ctrl_train': self.Y_control[train_ctrl_idx],
            'X_trt_train': X_trt_train,
            'Y_trt_train': self.Y_treatment[train_trt_idx],
            'X_trt_test': X_trt_test,
            'Y_trt_test': self.Y_treatment[test_trt_idx],
            'Y0_trt_train': self.Y0_treatment[train_trt_idx] if self.Y0_treatment is not None else None,
            'Y0_trt_test': self.Y0_treatment[test_trt_idx] if self.Y0_treatment is not None else None,
            'X0_train': X0_train,
            'X0_test': X0_test,
            'true_ATT': self.true_ATT,
            'train_ctrl_idx': train_ctrl_idx,
            'test_ctrl_idx': test_ctrl_idx,
            'train_trt_idx': train_trt_idx,
            'test_trt_idx': test_trt_idx,
            'transport_estimator': transport_estimator,
            'outcome_model': self.outcome_estimators[fold_idx]
        }

def bootstrap(rng, 
    data_dict: Dict,
    bootstrap_type: str = 'standard',
    stratified: bool = True,
) -> Dict:
    n0 = data_dict['n0']
    n1 = data_dict['n1']
    # Adjust sample sizes for subsampling
    if bootstrap_type == 'subsampling':
        # Use sqrt(n) samples - balances bias and variance
        m0 = int(np.sqrt(n0))
        m1 = int(np.sqrt(n1))
        if verbose:
            print(f"Subsampling: using {m0}/{n0} controls, {m1}/{n1} treated")
    else:
        m0, m1 = n0, n1

    # Create bootstrap sample
    if stratified:
        # Resample within treatment groups (RECOMMENDED for overlap issues)
        if bootstrap_type == 'bayesian':
            # Bayesian bootstrap: use Dirichlet weights
            weights0 = rng.dirichlet(np.ones(n0))
            weights1 = rng.dirichlet(np.ones(n1))
            
            # Convert to integer counts (approximate)
            idx0 = rng.choice(n0, size=n0, replace=True, p=weights0)
            idx1 = rng.choice(n1, size=n1, replace=True, p=weights1)
        else:
            # Standard or subsampling
            replace = (bootstrap_type != 'subsampling')
            idx0 = rng.choice(n0, size=m0, replace=replace)
            idx1 = rng.choice(n1, size=m1, replace=replace)
        
        # Create bootstrap data
        boot_data = {
            'X_control': data_dict['X_control'][idx0],
            'X_treatment': data_dict['X_treatment'][idx1],
            'Y_control': data_dict['Y_control'][idx0],
            'Y_treatment': data_dict['Y_treatment'][idx1],
            'Y0_treatment': None, 
            'true_ATT': data_dict['true_ATT'],
            'n0': len(idx0),
            'n1': len(idx1),
            'd': data_dict['d'],
            # 'trim_threshold': trim_threshold  # Pass to estimator
        }
        
    else:
        # Non-stratified: resample entire dataset
        # WARNING: Can worsen overlap issues!
        n_total = n0 + n1
        X_all = np.vstack([data_dict['X_control'], data_dict['X_treatment']])
        Y_all = np.concatenate([data_dict['Y_control'], data_dict['Y_treatment']])
        T_all = np.concatenate([np.zeros(n0), np.ones(n1)])
        
        idx = rng.choice(n_total, size=n_total, replace=True)
        
        X_boot = X_all[idx]
        Y_boot = Y_all[idx]
        T_boot = T_all[idx]
        
        boot_data = {
            'X_control': X_boot[T_boot == 0],
            'X_treatment': X_boot[T_boot == 1],
            'Y_control': Y_boot[T_boot == 0],
            'Y_treatment': Y_boot[T_boot == 1],
            'Y0_treatment': None,
            'true_ATT': data_dict['true_ATT'],
            'n0': int(np.sum(T_boot == 0)),
            'n1': int(np.sum(T_boot == 1)),
            'd': data_dict['d'],
            # 'trim_threshold': trim_threshold
        }

    return boot_data


