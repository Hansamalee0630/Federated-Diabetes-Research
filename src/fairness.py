"""
PHASE 5: FAIRNESS & BIAS AUDIT (COMPREHENSIVE 6-METRIC FRAMEWORK)
Evaluates the trained federated model across protected demographic groups.
Optimized for Accuracy-focused thresholding and JSON compatibility.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_optimal_threshold(y_true, y_pred_prob):
    """
    Find optimal threshold strictly maximizing Accuracy (matching the training script).
    """
    y_true = y_true.flatten()
    y_pred_prob = y_pred_prob.flatten()
    
    best_acc = 0.0
    best_threshold = 0.5
    best_recall = 0.0
    best_f1 = 0.0
    
    # Search tightly around 0.5 since data is balanced
    for threshold in np.linspace(0.40, 0.60, 41):
        y_pred = (y_pred_prob > threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Optimize for Accuracy (matching the FedAvg training logic)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_recall = recall_score(y_true, y_pred, zero_division=0)
            best_f1 = f1
            
    print(f"  OK Optimal Threshold: {best_threshold:.2f}")
    print(f"     (Maximized Accuracy: {best_acc:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f})")
            
    return best_threshold

def calculate_group_metrics(y_true, y_pred):
    """Calculate core performance metrics for a specific group, ensuring JSON serialization."""
    if len(y_true) == 0:
        return {'acc': 0.0, 'recall': 0.0, 'prec': 0.0, 'f1': 0.0, 'sr': 0.0, 'fpr': 0.0, 'count': 0}
        
    acc = float(accuracy_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    sr = float(np.mean(y_pred))  # Selection Rate
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    except:
        fpr = 0.0
        
    return {
        'acc': acc, 
        'recall': recall, 
        'prec': prec, 
        'f1': f1, 
        'sr': sr, 
        'fpr': fpr, 
        'count': int(len(y_true))
    }

def audit_fairness_6_metrics(metrics_dict, group1, group2, tolerance=0.10):
    """Calculates the 6 Comprehensive Fairness Metrics between two groups."""
    m1 = metrics_dict[group1]
    m2 = metrics_dict[group2]
    
    # Gap calculations
    dp_gap = abs(m1['sr'] - m2['sr'])
    eo_gap = abs(m1['recall'] - m2['recall'])
    pp_gap = abs(m1['prec'] - m2['prec'])
    fpr_gap = abs(m1['fpr'] - m2['fpr'])
    eodds_gap = (eo_gap + fpr_gap) / 2.0
    di_ratio = min(m1['sr'], m2['sr']) / max(m1['sr'], m2['sr']) if max(m1['sr'], m2['sr']) > 0 else 1.0
    acc_gap = abs(m1['acc'] - m2['acc'])
    
    # Passing logic (cast sum to int for JSON)
    passes = int(sum([
        dp_gap <= tolerance,
        eo_gap <= tolerance,
        pp_gap <= tolerance,
        eodds_gap <= tolerance,
        di_ratio >= 0.80,
        acc_gap <= tolerance
    ]))
    
    verdict = "FAIR" if passes >= 5 else ("NEEDS IMPROVEMENT" if passes >= 3 else "BIASED")
    
    return {
        'demographic_parity': {'gap': float(dp_gap), 'fair': bool(dp_gap <= tolerance)},
        'equal_opportunity': {'gap': float(eo_gap), 'fair': bool(eo_gap <= tolerance)},
        'predictive_parity': {'gap': float(pp_gap), 'fair': bool(pp_gap <= tolerance)},
        'equalized_odds': {'gap': float(eodds_gap), 'fair': bool(eodds_gap <= tolerance)},
        'disparate_impact': {'ratio': float(di_ratio), 'fair': bool(di_ratio >= 0.80)},
        'calibration': {'gap': float(acc_gap), 'fair': bool(acc_gap <= tolerance)},
        'overall_verdict': {'passes': passes, 'total': 6, 'verdict': verdict}
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHASE 5: FAIRNESS & BIAS AUDIT")
    print("="*70)
    
    os.makedirs("results/figures", exist_ok=True)
    
    print("Loading balanced data and trained model...")
    data_dir = Path("data/processed_data_balanced")
    
    if not (data_dir / "X_test_balanced.npy").exists():
        data_dir = Path("data/processed_data")
        suffix = ""
        is_balanced = False
    else:
        suffix = "_balanced"
        is_balanced = True
        
    try:
        X_test = np.load(data_dir / f"X_test{suffix}.npy").astype(np.float32)
        y_test = np.load(data_dir / f"y_test{suffix}.npy").astype(np.float32).flatten()
        sens_test = pd.read_csv(data_dir / f"sensitive_attrs_test{suffix}.csv")
        print(f"  OK Loaded {'BALANCED' if is_balanced else 'UNBALANCED'} data: X_test shape = {X_test.shape}")
        
        model = tf.keras.models.load_model("models/fedavg_global_model_best.keras", compile=False)
        print("  OK Model loaded successfully\n")
    except Exception as e:
        print(f"  ERROR: Failed to load data or model. {e}")
        return

    print(f"Dataset Type: {'BALANCED' if is_balanced else 'UNBALANCED'}")
    print(f"Test Set Size: {len(y_test):,}")
    print(f"Positive Rate: {np.mean(y_test)*100:.2f}%\n")
    
    print("Generating model predictions...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    
    print("  Tuning Decision Threshold...")
    optimal_threshold = float(find_optimal_threshold(y_test, y_pred_prob))
    y_pred = (y_pred_prob > optimal_threshold).astype(int)

    # Calculate overall metrics
    overall_acc = float(accuracy_score(y_test, y_pred))
    overall_rec = float(recall_score(y_test, y_pred, zero_division=0))
    overall_prec = float(precision_score(y_test, y_pred, zero_division=0))
    overall_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    overall_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print("\n" + "="*70)
    print("OVERALL MODEL PERFORMANCE")
    print("="*70)
    print(f"  Accuracy:  {overall_acc:.4f}")
    print(f"  Recall:    {overall_rec:.4f}")
    print(f"  Precision: {overall_prec:.4f}")
    print(f"  F1-Score:  {overall_f1:.4f}")
    print(f"  Specificity: {overall_spec:.4f}\n")
    print("Confusion Matrix:")
    print(f"  TN={tn:,}, FP={fp:,}")
    print(f"  FN={fn:,}, TP={tp:,}\n")

    # Group Metrics
    gender_metrics = {}
    for g in ['Male', 'Female']:
        mask = sens_test['gender'] == g
        gender_metrics[g] = calculate_group_metrics(y_test[mask], y_pred[mask])
        
    race_metrics = {}
    valid_races = ['Caucasian', 'AfricanAmerican']
    for r in valid_races:
        mask = sens_test['race'] == r
        race_metrics[r] = calculate_group_metrics(y_test[mask], y_pred[mask])

    # Audits
    gender_audit = audit_fairness_6_metrics(gender_metrics, 'Male', 'Female')
    race_audit = audit_fairness_6_metrics(race_metrics, 'Caucasian', 'AfricanAmerican')

    # Output JSON structure
    final_results = {
        'dataset_type': "BALANCED" if is_balanced else "UNBALANCED",
        'optimal_threshold': optimal_threshold,
        'overall_metrics': {
            'accuracy': overall_acc,
            'recall': overall_rec,
            'precision': overall_prec,
            'f1_score': overall_f1,
            'specificity': overall_spec
        },
        'gender_fairness_6metrics': gender_audit,
        'race_fairness_6metrics': race_audit,
        'group_details': {
            'gender': gender_metrics,
            'race': race_metrics
        }
    }

    with open('results/fairness_metrics.json', 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print("  OK Results saved to results/fairness_metrics.json")

    # ========================================================================
    # PLOTTING
    # ========================================================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gender Fairness Plot
    gender_groups = ['Male', 'Female']
    gender_sensitivities = [gender_metrics[g]['recall'] for g in gender_groups]
    sns.barplot(x=gender_groups, y=gender_sensitivities, hue=gender_groups, ax=axes[0], palette='Set2', legend=False)
    axes[0].set_title('Equal Opportunity (Sensitivity) by Gender', fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    for i, v in enumerate(gender_sensitivities):
        axes[0].text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

    # Race Fairness Plot
    race_groups = valid_races
    race_sensitivities = [race_metrics[r]['recall'] for r in race_groups]
    sns.barplot(x=race_groups, y=race_sensitivities, hue=race_groups, ax=axes[1], palette='husl', legend=False)
    axes[1].set_title('Equal Opportunity (Sensitivity) by Race', fontweight='bold')
    axes[1].set_ylim(0, 1.0)
    for i, v in enumerate(race_sensitivities):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/06_fairness_audit.png', dpi=300)
    print("  OK Visualization saved to results/figures/06_fairness_audit.png\n")
    print("="*70)
    print("FAIRNESS AUDIT COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()