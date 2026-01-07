"""
Utility functions for fairness-aware federated learning
"""

import numpy as np
import pandas as pd


def compute_adaptive_fairness_params(y_train, protected_attr, dataset_is_balanced):
    """
    Set fairness thresholds based on DATA PROPERTIES, not guessing
    
    For balanced datasets: Can afford stricter fairness constraints
    For imbalanced datasets: Must be more conservative
    
    Parameters:
    -----------
    y_train : array-like
        Training labels
    protected_attr : array-like
        Protected attribute values (e.g., gender)
    dataset_is_balanced : bool
        Whether dataset has been balanced
    
    Returns:
    --------
    dict with adaptive fairness parameters
    """
    
    # Analyze baseline fairness gap (before model training)
    baseline_gaps = {}
    for group in pd.Series(protected_attr).unique():
        mask = (pd.Series(protected_attr) == group).values
        baseline_gaps[group] = y_train[mask].mean()
    
    baseline_fairness_gap = max(baseline_gaps.values()) - min(baseline_gaps.values())
    
    if dataset_is_balanced:
        # Balanced data: data distribution is fair, 
        # so we can afford STRICTER fairness constraints
        fairness_threshold = baseline_fairness_gap * 0.5  # Reduce baseline gap by 50%
        fairness_lambda = 0.1  # Higher weight on fairness (10% of loss)
        strategy = 'strict'
    else:
        # Imbalanced data: harder to achieve fairness
        # Settle for gradual improvement
        fairness_threshold = baseline_fairness_gap * 0.8  # Keep 80% of baseline
        fairness_lambda = 0.03  # Lower weight on fairness (3% of loss)
        strategy = 'conservative'
    
    return {
        'threshold': fairness_threshold,
        'lambda': fairness_lambda,
        'baseline_gap': baseline_fairness_gap,
        'strategy': strategy,
        'baseline_gaps': baseline_gaps
    }


def compute_federated_convergence_criterion(history, window=5):
    """
    FL convergence = both accuracy AND fairness stabilized
    
    Parameters:
    -----------
    history : dict
        Training history with 'f1', 'fairness_gap', etc.
    window : int
        Window size for convergence check
    
    Returns:
    --------
    bool : True if converged, False otherwise
    """
    
    if len(history.get('f1', [])) < window:
        return False
    
    # 1. Accuracy stability (CV < 2%)
    recent_f1s = np.array(history['f1'][-window:])
    f1_cv = np.std(recent_f1s) / np.mean(recent_f1s)
    
    # 2. Fairness stability (gap changes < 5%)
    recent_gaps = np.array(history.get('fairness_gap', [1.0] * window)[-window:])
    gap_change = np.std(np.diff(recent_gaps)) if len(recent_gaps) > 1 else 0.0
    
    # 3. Both stable = convergence
    has_converged = (f1_cv < 0.02) and (gap_change < 0.05)
    
    return has_converged


if __name__ == "__main__":
    # Example usage
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    protected_attr = np.array(['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'])
    
    params_balanced = compute_adaptive_fairness_params(y_train, protected_attr, True)
    print("Adaptive Fairness Parameters (Balanced Dataset):")
    print(params_balanced)
    
    params_imbalanced = compute_adaptive_fairness_params(y_train, protected_attr, False)
    print("\nAdaptive Fairness Parameters (Imbalanced Dataset):")
    print(params_imbalanced)
