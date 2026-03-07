import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import json
from scipy.stats import wasserstein_distance
import warnings

warnings.filterwarnings('ignore')

def load_balanced_train_data():
    """Load balanced TRAINING data for Non-IID analysis"""
    print("Loading balanced TRAINING data for comprehensive Non-IID analysis...")
    
    data_dir = Path('data/processed_data_balanced')
    
    # Fallback to standard processed data folder if not found
    if not (data_dir / 'X_train_balanced.npy').exists():
        data_dir = Path('data/processed_data')
        
    x_file = 'X_train_balanced.npy' if (data_dir / 'X_train_balanced.npy').exists() else 'X_train.npy'
    y_file = 'y_train_balanced.npy' if (data_dir / 'y_train_balanced.npy').exists() else 'y_train.npy'
    sens_file = 'sensitive_attrs_train_balanced.csv' if (data_dir / 'sensitive_attrs_train_balanced.csv').exists() else 'sensitive_attrs_train.csv'
    
    X_train = np.load(data_dir / x_file).astype(np.float32)
    y_train = np.load(data_dir / y_file).astype(np.float32)
    sens_train = pd.read_csv(data_dir / sens_file)
    
    print(f" ✓ Data loaded: {X_train.shape}")
    return X_train, y_train, sens_train

def comprehensive_non_iid_analysis(X_train, y_train, sens_train):
    print("\n" + "="*70)
    print("COMPREHENSIVE NON-IID QUANTIFICATION (6 METRICS)")
    print("="*70)
    
    hospitals = [1, 2, 3]
    hospital_data = {h: {} for h in hospitals}
    
    # Partition by hospital
    for h in hospitals:
        mask = (sens_train['hospital_id'] == h).values
        hospital_data[h]['X'] = X_train[mask]
        hospital_data[h]['y'] = y_train[mask]
        hospital_data[h]['n_samples'] = len(X_train[mask])
    
    results = {}
    
    # ====================================================================
    # 1. LABEL HETEROGENEITY (Should be low now due to balancing)
    # ====================================================================
    print("\n1. LABEL HETEROGENEITY (Label Distribution Shift)")
    label_hets = {}
    global_label_dist = np.bincount(y_train.astype(int), minlength=2) / len(y_train)
    
    for h in hospitals:
        y_h = hospital_data[h]['y']
        local_label_dist = np.bincount(y_h.astype(int), minlength=2) / len(y_h)
        lh = jensenshannon(local_label_dist, global_label_dist)
        label_hets[h] = lh
        print(f"   Hospital {h}: Pos rate = {y_h.mean()*100:.2f}%, JS Divergence = {lh:.4f}")
    
    results['label_heterogeneity'] = {'average': float(np.mean(list(label_hets.values())))}
    
    # ====================================================================
    # 2. FEATURE HETEROGENEITY (Covariate Shift)
    # ====================================================================
    print("\n2. FEATURE HETEROGENEITY (Feature Distribution Variation)")
    feature_hets = {}
    for h in hospitals:
        X_h = hospital_data[h]['X']
        feature_means = np.abs(X_h.mean(axis=0)) + 1e-8
        feature_stds = X_h.std(axis=0)
        cv = feature_stds / feature_means
        feature_hets[h] = float(np.nanmean(cv))
        print(f"   Hospital {h}: Mean feature Coefficient of Variation = {feature_hets[h]:.4f}")
    
    results['feature_heterogeneity'] = {'average': float(np.mean(list(feature_hets.values())))}
    
    # ====================================================================
    # 3. CORRELATION HETEROGENEITY
    # ====================================================================
    print("\n3. CORRELATION HETEROGENEITY (Feature Relationship Variation)")
    correlation_hets = {}
    for h in hospitals:
        X_h = hospital_data[h]['X']
        X_sample = X_h[:, :min(20, X_h.shape[1])]  # Top 20 features
        if len(X_sample) > 10:
            corr_matrix = np.corrcoef(X_sample.T)
            ch = float(np.linalg.norm(np.nan_to_num(corr_matrix)))
            correlation_hets[h] = ch
            print(f"   Hospital {h}: Correlation Frobenius norm = {ch:.4f}")
            
    results['correlation_heterogeneity'] = {'average': float(np.mean(list(correlation_hets.values())))}
    
    # ====================================================================
    # 4. CLASS IMBALANCE HETEROGENEITY
    # ====================================================================
    print("\n4. CLASS IMBALANCE HETEROGENEITY (Imbalance Variation)")
    class_imbalance_hets = {}
    global_pos_rate = y_train.mean()
    for h in hospitals:
        y_h = hospital_data[h]['y']
        local_pos_rate = y_h.mean()
        cih = abs(local_pos_rate - global_pos_rate)
        class_imbalance_hets[h] = cih
        print(f"   Hospital {h}: Diff from global positive rate = {cih*100:.2f}%")
        
    results['class_imbalance_heterogeneity'] = {'max_diff': float(max(class_imbalance_hets.values()))}
    
    # ====================================================================
    # 5. COVARIATE SHIFT INDEX
    # ====================================================================
    print("\n5. COVARIATE SHIFT INDEX (Mean Feature Discrepancy)")
    csi_values = {}
    for i, h1 in enumerate(hospitals):
        for h2 in hospitals[i+1:]:
            X_h1 = hospital_data[h1]['X'][:min(500, hospital_data[h1]['n_samples'])]
            X_h2 = hospital_data[h2]['X'][:min(500, hospital_data[h2]['n_samples'])]
            mfd = float(np.linalg.norm(X_h1.mean(axis=0) - X_h2.mean(axis=0)))
            csi_values[f'H{h1}_vs_H{h2}'] = mfd
            print(f"   Hospital {h1} vs {h2}: Mean Feature Discrepancy = {mfd:.6f}")
            
    results['covariate_shift_index'] = {'average': float(np.mean(list(csi_values.values())))}
    
    # ====================================================================
    # 6. WASSERSTEIN DISTANCE
    # ====================================================================
    print("\n6. WASSERSTEIN DISTANCE (Label Distribution)")
    wd_values = {}
    for i, h1 in enumerate(hospitals):
        for h2 in hospitals[i+1:]:
            y_h1 = hospital_data[h1]['y']
            y_h2 = hospital_data[h2]['y']
            wd = float(wasserstein_distance(
                np.bincount(y_h1.astype(int), minlength=2) / len(y_h1),
                np.bincount(y_h2.astype(int), minlength=2) / len(y_h2)
            ))
            wd_values[f'H{h1}_vs_H{h2}'] = wd
            print(f"   Hospital {h1} vs {h2}: Wasserstein Dist = {wd:.6f}")
            
    results['wasserstein_distance'] = {'average': float(np.mean(list(wd_values.values())))}
    
    # ====================================================================
    # COMPOSITE SCORE & REPORT
    # ====================================================================
    # Aggregated metric mapping
    composite_score = (
        0.20 * results['label_heterogeneity']['average'] +
        0.15 * np.tanh(results['feature_heterogeneity']['average']) +
        0.10 * np.tanh(results['correlation_heterogeneity']['average'] / 10) +
        0.25 * results['class_imbalance_heterogeneity']['max_diff'] +
        0.15 * np.tanh(results['covariate_shift_index']['average']) +
        0.15 * results['wasserstein_distance']['average']
    )
    
    print("\n" + "="*70)
    print("NON-IID QUANTIFICATION SUMMARY")
    print("="*70)
    print(f"\nComposite Non-IID Score: {composite_score:.4f}")
    
    if composite_score < 0.15: severity = 'MINIMAL (Almost IID)'
    elif composite_score < 0.30: severity = 'LOW'
    elif composite_score < 0.50: severity = 'MODERATE (Realistic Federated Environment)'
    else: severity = 'HIGH'
    
    print(f"Severity Level: {severity}")

def analyze_non_iid():
    X_train, y_train, sens_train = load_balanced_train_data()
    comprehensive_non_iid_analysis(X_train, y_train, sens_train)

if __name__ == "__main__":
    analyze_non_iid()