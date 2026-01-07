import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, 
    precision_score, f1_score
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_data_and_model():
    """Load balanced data and trained model"""
    print("Loading balanced data and trained model...")
    
    # CRITICAL: Use balanced data path
    data_dir = Path('data/processed_data_balanced')
    
    # Fallback to unbalanced if needed (for development only)
    if not (data_dir / 'X_test_balanced.npy').exists():
        print("\n  Balanced data not found. Falling back to unbalanced data.")
        data_dir = Path('data/processed_data')
        X_test = np.load(data_dir / 'X_test.npy').astype(np.float32)
        y_test = np.load(data_dir / 'y_test.npy').astype(np.float32)
        sens_test = pd.read_csv(data_dir / 'sensitive_attrs_test.csv')
        dataset_type = "UNBALANCED"
    else:
        X_test = np.load(data_dir / 'X_test_balanced.npy').astype(np.float32)
        y_test = np.load(data_dir / 'y_test_balanced.npy').astype(np.float32)
        sens_test = pd.read_csv(data_dir / 'sensitive_attrs_test_balanced.csv')
        dataset_type = "BALANCED"
    
    print(f" OK Loaded {dataset_type} data: X_test shape = {X_test.shape}")
    
    # Load model
    model_path = 'models/fedavg_global_model_best.keras'
    if not os.path.exists(model_path):
        print(f" Error: Model not found at {model_path}")
        print(" Run fedavg_training.py first")
        exit()
        
    model = tf.keras.models.load_model(model_path, compile=False)
    print(" OK Model loaded successfully")
    
    return X_test, y_test, sens_test, model, dataset_type

def find_optimal_threshold(y_true, y_pred_prob, min_recall=0.55):
    """
    Find decision threshold that maximizes F1 while maintaining min recall
    """
    best_f1 = 0
    best_thresh = 0.5
    best_recall = 0.0
    
    y_true = y_true.flatten()
    y_pred_prob = y_pred_prob.flatten()
    
    print("\n Tuning Decision Threshold...")
    
    for thresh in np.arange(0.05, 0.55, 0.01):
        y_pred = (y_pred_prob > thresh).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        if recall >= min_recall:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_recall = recall
    
    if best_f1 == 0:
        # Fallback if min_recall not met
        best_thresh = 0.5
    
    print(f" OK Optimal Threshold: {best_thresh:.2f}")
    print(f"   (F1-Score: {best_f1:.4f}, Recall: {best_recall:.4f})")
    
    return best_thresh

def six_metric_fairness_audit(y_true, y_pred_prob, protected_attr, 
                              protected_value, threshold=0.5):
    """
    Comprehensive fairness evaluation (6 metrics)
    
    Based on:
    - Nature Health Systems (2025): Fairness in predictive healthcare
    - FairFML framework: Multi-metric fairness definitions
    - US EEOC (Equal Employment): 4/5 rule (disparate impact)
    
    Returns dict with 6 fairness metrics and verdicts
    """
    
    y_pred = (y_pred_prob.flatten() > threshold).astype(int)
    y_true_flat = y_true.flatten().astype(int)
    
    # Split by protected attribute
    prot_mask = (protected_attr.values == protected_value)
    rest_mask = ~prot_mask
    
    results = {}
    
    # ====================================================================
    # METRIC 1: DEMOGRAPHIC PARITY
    # ====================================================================
    prot_pos_rate = y_pred[prot_mask].mean()
    rest_pos_rate = y_pred[rest_mask].mean()
    dp_gap = abs(prot_pos_rate - rest_pos_rate)
    
    results['demographic_parity'] = {
        'metric_name': 'Demographic Parity',
        'gap': float(dp_gap),
        'protected_group_rate': float(prot_pos_rate),
        'other_group_rate': float(rest_pos_rate),
        'fair': dp_gap < 0.10,
        'fair_threshold': 0.10,
        'clinical_meaning': 'Should recommend readmission prevention to both groups equally',
        'fairness_level': (
            'FAIR' if dp_gap < 0.10 else
            'WARNING' if dp_gap < 0.15 else
            'BIASED'
        )
    }
    
    # ====================================================================
    # METRIC 2: EQUAL OPPORTUNITY (PRIMARY FOR HEALTHCARE)
    # ====================================================================
    tpr_prot = recall_score(y_true_flat[prot_mask], y_pred[prot_mask], zero_division=0)
    tpr_rest = recall_score(y_true_flat[rest_mask], y_pred[rest_mask], zero_division=0)
    eo_gap = abs(tpr_prot - tpr_rest)
    
    results['equal_opportunity'] = {
        'metric_name': 'Equal Opportunity (Sensitivity)',
        'gap': float(eo_gap),
        'protected_group_sensitivity': float(tpr_prot),
        'other_group_sensitivity': float(tpr_rest),
        'fair': eo_gap < 0.10,
        'fair_threshold': 0.10,
        'clinical_meaning': '[CRITICAL] Equal ability to detect true readmission cases',
        'fairness_level': (
            'FAIR' if eo_gap < 0.10 else
            'WARNING' if eo_gap < 0.15 else
            'BIASED - CLINICAL CONCERN'
        ),
        'priority': 'HIGHEST'
    }
    
    # ====================================================================
    # METRIC 3: PREDICTIVE PARITY (Precision)
    # ====================================================================
    ppv_prot = precision_score(y_true_flat[prot_mask], y_pred[prot_mask], zero_division=0)
    ppv_rest = precision_score(y_true_flat[rest_mask], y_pred[rest_mask], zero_division=0)
    pp_gap = abs(ppv_prot - ppv_rest)
    
    results['predictive_parity'] = {
        'metric_name': 'Predictive Parity (Precision)',
        'gap': float(pp_gap),
        'protected_group_precision': float(ppv_prot),
        'other_group_precision': float(ppv_rest),
        'fair': pp_gap < 0.10,
        'fair_threshold': 0.10,
        'clinical_meaning': 'When flagged as high-risk, both groups should actually be at risk',
        'fairness_level': (
            'FAIR' if pp_gap < 0.10 else
            'WARNING' if pp_gap < 0.15 else
            'BIASED'
        )
    }
    
    # ====================================================================
    # METRIC 4: EQUALIZED ODDS
    # ====================================================================
    y_true_neg = 1 - y_true_flat
    y_pred_neg = 1 - y_pred
    
    tnr_prot = (y_pred_neg[prot_mask] & y_true_neg[prot_mask]).sum() / max((y_true_neg[prot_mask]).sum(), 1)
    tnr_rest = (y_pred_neg[rest_mask] & y_true_neg[rest_mask]).sum() / max((y_true_neg[rest_mask]).sum(), 1)
    tnr_gap = abs(tnr_prot - tnr_rest)
    
    eq_odds_gap = (eo_gap + tnr_gap) / 2
    
    results['equalized_odds'] = {
        'metric_name': 'Equalized Odds',
        'tpr_gap': float(eo_gap),
        'tnr_gap': float(tnr_gap),
        'combined_gap': float(eq_odds_gap),
        'protected_tpr': float(tpr_prot),
        'protected_tnr': float(tnr_prot),
        'other_tpr': float(tpr_rest),
        'other_tnr': float(tnr_rest),
        'fair': eq_odds_gap < 0.10,
        'fair_threshold': 0.10,
        'clinical_meaning': 'Both error rates should be equal across groups',
        'fairness_level': (
            'FAIR' if eq_odds_gap < 0.10 else
            'WARNING' if eq_odds_gap < 0.15 else
            'BIASED'
        )
    }
    
    # ====================================================================
    # METRIC 5: DISPARATE IMPACT RATIO (4/5 Rule)
    # ====================================================================
    selection_rates = [prot_pos_rate, rest_pos_rate]
    dir_ratio = min(selection_rates) / max(selection_rates) if max(selection_rates) > 0 else 1.0
    
    results['disparate_impact_ratio'] = {
        'metric_name': 'Disparate Impact Ratio (4/5 Rule)',
        'ratio': float(dir_ratio),
        'fair': dir_ratio >= 0.8,
        'fair_threshold': 0.80,
        'legal_standard': 'US EEOC (Equal Employment Opportunity Commission)',
        'clinical_meaning': 'Protected group selection rate >= 80% of majority group',
        'fairness_level': (
            'FAIR' if dir_ratio >= 0.8 else
            'WARNING' if dir_ratio >= 0.6 else
            'POTENTIAL DISCRIMINATION'
        )
    }
    
    # ====================================================================
    # METRIC 6: CALIBRATION (Accuracy within groups)
    # ====================================================================
    acc_prot = accuracy_score(y_true_flat[prot_mask], y_pred[prot_mask])
    acc_rest = accuracy_score(y_true_flat[rest_mask], y_pred[rest_mask])
    calib_gap = abs(acc_prot - acc_rest)
    
    results['calibration'] = {
        'metric_name': 'Calibration (Accuracy Balance)',
        'gap': float(calib_gap),
        'protected_accuracy': float(acc_prot),
        'other_accuracy': float(acc_rest),
        'fair': calib_gap < 0.05,
        'fair_threshold': 0.05,
        'clinical_meaning': 'Model should be equally accurate for both groups overall',
        'fairness_level': (
            'FAIR' if calib_gap < 0.05 else
            'WARNING' if calib_gap < 0.10 else
            'BIASED'
        ),
        'note': 'Strictest threshold (0.05) for clinical decisions'
    }
    
    # ====================================================================
    # OVERALL FAIRNESS VERDICT
    # ====================================================================
    fair_count = sum(1 for m in results.values() if m.get('fair', False))
    total_metrics = len(results)
    
    results['overall_verdict'] = {
        'metrics_passing': fair_count,
        'total_metrics': total_metrics,
        'verdict': (
            'FAIR - All metrics pass' if fair_count == total_metrics else
            'ACCEPTABLE - Most metrics pass' if fair_count >= 4 else
            'CONCERNING - Multiple fairness issues' if fair_count >= 2 else
            'BIASED - Systemic fairness problems'
        ),
        'primary_concern': 'EQUAL_OPPORTUNITY' if not results['equal_opportunity']['fair'] else None,
        'recommendation': (
            'Model ready for deployment' if fair_count == total_metrics else
            'Recommend bias mitigation before deployment' if fair_count < 4 else
            'Monitor fairness in production'
        )
    }
    
    return results


def calculate_fairness_metrics(y_true, y_pred_prob, sensitive_array, 
                                threshold, group_name):
    """
    Calculate per-group metrics using optimal threshold
    (Legacy function for visualization compatibility)
    """
    y_pred = (y_pred_prob.flatten() > threshold).astype(int)
    y_true = y_true.flatten().astype(int)
    
    df = pd.DataFrame({
        'y_true': y_true, 
        'y_pred': y_pred, 
        'group': sensitive_array
    })
    
    metrics = []
    
    for group in df['group'].unique():
        sub_df = df[df['group'] == group]
        
        # Skip small groups
        if len(sub_df) < 20:
            continue
        
        acc = accuracy_score(sub_df['y_true'], sub_df['y_pred'])
        recall = recall_score(sub_df['y_true'], sub_df['y_pred'], zero_division=0)
        precision = precision_score(sub_df['y_true'], sub_df['y_pred'], zero_division=0)
        f1 = f1_score(sub_df['y_true'], sub_df['y_pred'], zero_division=0)
        
        metrics.append({
            'Group': group, 
            'Accuracy': acc, 
            'Recall': recall, 
            'Precision': precision,
            'F1-Score': f1,
            'Count': len(sub_df)
        })
    
    metrics_df = pd.DataFrame(metrics).sort_values('Recall', ascending=False)
    
    # Calculate fairness gaps
    if len(metrics_df) > 0:
        acc_gap = metrics_df['Accuracy'].max() - metrics_df['Accuracy'].min()
        recall_gap = metrics_df['Recall'].max() - metrics_df['Recall'].min()
        f1_gap = metrics_df['F1-Score'].max() - metrics_df['F1-Score'].min()
    else:
        acc_gap = recall_gap = f1_gap = 0.0
    
    return metrics_df, acc_gap, recall_gap, f1_gap

def analyze_fairness():
    """Run complete fairness audit"""
    print("\n" + "="*70)
    print("PHASE 5: FAIRNESS & BIAS AUDIT")
    print("="*70)
    
    X_test, y_test, sens_test, model, dataset_type = load_data_and_model()
    
    print(f"\nDataset Type: {dataset_type}")
    print(f"Test Set Size: {len(y_test):,}")
    print(f"Positive Rate: {y_test.mean()*100:.2f}%")
    
    # Generate predictions
    print("\nGenerating model predictions...")
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    
    # Find optimal threshold
    optimal_thresh = find_optimal_threshold(y_test, y_pred_prob, min_recall=0.55)
    y_pred_binary = (y_pred_prob > optimal_thresh).astype(int)
    
    # Overall performance
    print("\n" + "="*70)
    print("OVERALL MODEL PERFORMANCE")
    print("="*70)
    
    y_test_flat = y_test.flatten().astype(int)
    
    overall_acc = accuracy_score(y_test_flat, y_pred_binary)
    overall_recall = recall_score(y_test_flat, y_pred_binary, zero_division=0)
    overall_precision = precision_score(y_test_flat, y_pred_binary, zero_division=0)
    overall_f1 = f1_score(y_test_flat, y_pred_binary, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_binary, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"  Accuracy:  {overall_acc:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  F1-Score:  {overall_f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:,}, FP={fp:,}")
    print(f"  FN={fn:,}, TP={tp:,}")
    
    # ====================================================================
    # GENDER FAIRNESS (PRIMARY) - 6-METRIC AUDIT
    # ====================================================================
    print("\n" + "="*70)
    print("GENDER FAIRNESS (PRIMARY - 6-METRIC COMPREHENSIVE AUDIT)")
    print("="*70)
    
    gender_fairness = six_metric_fairness_audit(
        y_test, y_pred_prob, sens_test['gender'], 
        'Female', threshold=optimal_thresh
    )
    
    print("\nGENDER FAIRNESS AUDIT RESULTS:")
    for metric_name, metric_data in gender_fairness.items():
        if metric_name == 'overall_verdict':
            continue
        
        print(f"\n  {metric_data['metric_name'].upper()}")
        if 'ratio' in metric_data:
            print(f"    Ratio: {metric_data['ratio']:.3f} | Fair: {metric_data['fair']}")
        elif 'combined_gap' in metric_data:
            print(f"    Combined Gap: {metric_data['combined_gap']*100:.2f}% | Fair: {metric_data['fair']}")
        elif 'gap' in metric_data:
            print(f"    Gap: {metric_data['gap']*100:.2f}% | Fair: {metric_data['fair']}")
        print(f"    --> {metric_data['clinical_meaning']}")
        print(f"    Level: {metric_data['fairness_level']}")
    
    print(f"\n  OVERALL VERDICT:")
    print(f"    {gender_fairness['overall_verdict']['verdict']}")
    print(f"    ({gender_fairness['overall_verdict']['metrics_passing']}/{gender_fairness['overall_verdict']['total_metrics']} metrics pass)")
    if gender_fairness['overall_verdict']['recommendation']:
        print(f"    --> {gender_fairness['overall_verdict']['recommendation']}")
    
    # ====================================================================
    # RACE FAIRNESS (SECONDARY) - 6-METRIC AUDIT
    # ====================================================================
    print("\n" + "="*70)
    print("RACE FAIRNESS (SECONDARY - 6-METRIC COMPREHENSIVE AUDIT)")
    print("="*70)
    
    # For race, use majority group as protected for consistency
    race_fairness = six_metric_fairness_audit(
        y_test, y_pred_prob, sens_test['race'], 
        sens_test['race'].mode()[0] if len(sens_test['race'].mode()) > 0 else 'Caucasian',
        threshold=optimal_thresh
    )
    
    print("\nRACE FAIRNESS AUDIT RESULTS:")
    for metric_name, metric_data in race_fairness.items():
        if metric_name == 'overall_verdict':
            continue
        
        print(f"\n  {metric_data['metric_name'].upper()}")
        if 'ratio' in metric_data:
            print(f"    Ratio: {metric_data['ratio']:.3f} | Fair: {metric_data['fair']}")
        elif 'combined_gap' in metric_data:
            print(f"    Combined Gap: {metric_data['combined_gap']*100:.2f}% | Fair: {metric_data['fair']}")
        elif 'gap' in metric_data:
            print(f"    Gap: {metric_data['gap']*100:.2f}% | Fair: {metric_data['fair']}")
        print(f"    --> {metric_data['clinical_meaning']}")
        print(f"    Level: {metric_data['fairness_level']}")
    
    print(f"\n  OVERALL VERDICT:")
    print(f"    {race_fairness['overall_verdict']['verdict']}")
    print(f"    ({race_fairness['overall_verdict']['metrics_passing']}/{race_fairness['overall_verdict']['total_metrics']} metrics pass)")
    if race_fairness['overall_verdict']['recommendation']:
        print(f"    --> {race_fairness['overall_verdict']['recommendation']}")
    
    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    fairness_results = {
        'dataset_type': dataset_type,
        'optimal_threshold': float(optimal_thresh),
        'overall_metrics': {
            'accuracy': float(overall_acc),
            'recall': float(overall_recall),
            'precision': float(overall_precision),
            'f1_score': float(overall_f1),
            'specificity': float(specificity)
        },
        'gender_fairness_6metrics': gender_fairness,
        'race_fairness_6metrics': race_fairness
    }
    
    # Ensure all numpy / pandas types are converted to native Python types
    def make_json_serializable(obj):
        """Recursively convert numpy/pandas types to native Python types for JSON."""
        # dict
        if isinstance(obj, dict):
            return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
        # list/tuple
        if isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        # pandas Series
        try:
            import pandas as _pd
            if isinstance(obj, _pd.Series):
                return make_json_serializable(obj.tolist())
        except Exception:
            pass
        # numpy arrays and scalars
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray):
                return make_json_serializable(obj.tolist())
            if isinstance(obj, (_np.generic,)):
                return obj.item()
        except Exception:
            pass
        # fallback (native python types remain unchanged)
        return obj

    serializable_results = make_json_serializable(fairness_results)
    with open('results/fairness_metrics.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\nOK Results saved to results/fairness_metrics.json")
    
    # ====================================================================
    # VISUALIZATIONS
    # ====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract fairness metrics for visualization
    gender_eo_gap = gender_fairness['equal_opportunity']['gap']
    race_eo_gap = race_fairness['equal_opportunity']['gap']
    
    # Gender Fairness - Equal Opportunity
    gender_groups = ['Female', 'Other']
    gender_sensitivities = [
        gender_fairness['equal_opportunity']['protected_group_sensitivity'],
        gender_fairness['equal_opportunity']['other_group_sensitivity']
    ]
    sns.barplot(
        x=gender_groups, y=gender_sensitivities, 
        ax=axes[0, 0], palette='Set2', hue=gender_groups, legend=False
    )
    axes[0, 0].set_title(
        f'Gender Fairness - Equal Opportunity (Sensitivity)\n(Gap: {gender_eo_gap*100:.1f}%)',
        fontweight='bold', fontsize=12
    )
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel('Sensitivity (TPR)')
    axes[0, 0].set_xlabel('Group')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Race Fairness - Equal Opportunity
    race_protected = sens_test['race'].mode()[0] if len(sens_test['race'].mode()) > 0 else 'Caucasian'
    race_groups = [race_protected, 'Other']
    race_sensitivities = [
        race_fairness['equal_opportunity']['protected_group_sensitivity'],
        race_fairness['equal_opportunity']['other_group_sensitivity']
    ]
    sns.barplot(
        x=race_groups, y=race_sensitivities, 
        ax=axes[0, 1], palette='husl', hue=race_groups, legend=False
    )
    axes[0, 1].set_title(
        f'Race Fairness - Equal Opportunity (Sensitivity)\n(Gap: {race_eo_gap*100:.1f}%)',
        fontweight='bold', fontsize=12
    )
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_ylabel('Sensitivity (TPR)')
    axes[0, 1].set_xlabel('Group')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_flat, y_pred_binary, labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
        cbar=False, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes']
    )
    axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Metrics Summary
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'Specificity'],
        'Value': [overall_acc, overall_recall, overall_precision, overall_f1, specificity]
    })
    
    axes[1, 1].axis('off')
    table = axes[1, 1].table(
        cellText=metrics_summary.values,
        colLabels=metrics_summary.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Overall Model Metrics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/figures/06_fairness_audit.png', dpi=300, bbox_inches='tight')
    print("OK Visualization saved to results/figures/06_fairness_audit.png")
    plt.close()
    
    print("\n" + "="*70)
    print("FAIRNESS AUDIT COMPLETE")
    print("="*70)

if __name__ == "__main__":
    analyze_fairness()