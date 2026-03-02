import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import json
from scipy.stats import entropy

def load_balanced_train_data():
    """Load balanced TRAINING data for Non-IID analysis"""
    print("Loading balanced TRAINING data for comprehensive Non-IID analysis...")
    
    data_dir = Path('data/processed_data_balanced')
    
    if not (data_dir / 'X_train_balanced.npy').exists():
        print("\n  Balanced data not found. Falling back to unbalanced data.")
        data_dir = Path('data/processed_data')
        X_train = np.load(data_dir / 'X_train.npy').astype(np.float32)
        y_train = np.load(data_dir / 'y_train.npy').astype(np.float32)
        sens_train = pd.read_csv(data_dir / 'sensitive_attrs_train.csv')
        is_balanced = False
    else:
        X_train = np.load(data_dir / 'X_train_balanced.npy').astype(np.float32)
        y_train = np.load(data_dir / 'y_train_balanced.npy').astype(np.float32)
        sens_train = pd.read_csv(data_dir / 'sensitive_attrs_train_balanced.csv')
        is_balanced = True
    
    print(f" ✓ Data loaded: {X_train.shape}")
    print(f" ✓ Type: {'BALANCED' if is_balanced else 'UNBALANCED'}")
    
    return X_train, y_train, sens_train, is_balanced


def comprehensive_non_iid_analysis(X_train, y_train, sens_train, feature_names):
    """
    6-metric Non-IID assessment per 2024 literature
    
    Metrics:
    1. Label Heterogeneity (LH)
    2. Feature Heterogeneity (FH)
    3. Correlation Heterogeneity (CH)
    4. Class Imbalance Heterogeneity (CIH)
    5. Covariate Shift Index (CSI)
    6. Wasserstein Distance (WD)
    """
    
    print("\n" + "="*70)
    print("COMPREHENSIVE NON-IID QUANTIFICATION (6 METRICS)")
    print("="*70)
    
    from scipy.stats import wasserstein_distance
    
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
    # METRIC 1: LABEL HETEROGENEITY
    # ====================================================================
    print("\n1. LABEL HETEROGENEITY (Label Distribution Shift)")
    print("   How different are positive class rates across hospitals?")
    
    label_hets = {}
    global_label_dist = np.bincount(y_train.astype(int), minlength=2) / len(y_train)
    
    for h in hospitals:
        y_h = hospital_data[h]['y']
        local_label_dist = np.bincount(y_h.astype(int), minlength=2) / len(y_h)
        
        # Jensen-Shannon divergence (symmetric KL divergence)
        lh = jensenshannon(local_label_dist, global_label_dist)
        label_hets[h] = lh
        
        print(f"   Hospital {h}: Pos rate = {y_h.mean()*100:.2f}%, JS = {lh:.4f}")
    
    results['label_heterogeneity'] = {
        'per_hospital': label_hets,
        'average': float(np.mean(list(label_hets.values()))),
        'std': float(np.std(list(label_hets.values()))),
        'global_positive_rate': float(global_label_dist[1]),
        'interpretation': (
            'HIGH' if np.mean(list(label_hets.values())) > 0.2 else
            'MODERATE' if np.mean(list(label_hets.values())) > 0.1 else
            'LOW'
        )
    }
    
    # ====================================================================
    # METRIC 2: FEATURE HETEROGENEITY
    # ====================================================================
    print("\n2. FEATURE HETEROGENEITY (Feature Distribution Variation)")
    print("   How different are feature distributions across hospitals?")
    
    feature_hets = {}
    for h in hospitals:
        X_h = hospital_data[h]['X']
        
        # Coefficient of variation per feature (std / mean)
        feature_means = np.abs(X_h.mean(axis=0)) + 1e-8
        feature_stds = X_h.std(axis=0)
        
        cv = feature_stds / feature_means
        feature_hets[h] = float(np.nanmean(cv))
        
        print(f"   Hospital {h}: Mean feature CV = {feature_hets[h]:.4f}")
    
    results['feature_heterogeneity'] = {
        'per_hospital': feature_hets,
        'average': float(np.mean(list(feature_hets.values()))),
        'interpretation': 'Features normalized; high CV suggests outliers'
    }
    
    # ====================================================================
    # METRIC 3: CORRELATION HETEROGENEITY
    # ====================================================================
    print("\n3. CORRELATION HETEROGENEITY (Feature Relationship Variation)")
    print("   Do features correlate differently across hospitals?")
    
    correlation_hets = {}
    for h in hospitals:
        X_h = hospital_data[h]['X']
        # Sample first 20 features for correlation
        X_sample = X_h[:, :min(20, X_h.shape[1])]
        
        if len(X_sample) > 10:
            corr_matrix = np.corrcoef(X_sample.T)
            # Frobenius norm (total correlation strength)
            ch = float(np.linalg.norm(np.nan_to_num(corr_matrix)))
            correlation_hets[h] = ch
            
            print(f"   Hospital {h}: Correlation norm = {ch:.4f}")
    
    results['correlation_heterogeneity'] = {
        'per_hospital': correlation_hets,
        'average': float(np.mean(list(correlation_hets.values()))),
        'interpretation': 'Higher = stronger inter-feature correlations'
    }
    
    # ====================================================================
    # METRIC 4: CLASS IMBALANCE HETEROGENEITY
    # ====================================================================
    print("\n4. CLASS IMBALANCE HETEROGENEITY (Imbalance Variation)")
    print("   How different are class balances across hospitals?")
    
    class_imbalance_hets = {}
    global_pos_rate = y_train.mean()
    
    for h in hospitals:
        y_h = hospital_data[h]['y']
        local_pos_rate = y_h.mean()
        cih = abs(local_pos_rate - global_pos_rate)
        class_imbalance_hets[h] = cih
        
        print(f"   Hospital {h}: Pos rate = {local_pos_rate*100:.2f}%, Diff = {cih*100:.2f}%")
    
    results['class_imbalance_heterogeneity'] = {
        'per_hospital': class_imbalance_hets,
        'max_diff': float(max(class_imbalance_hets.values())),
        'global_rate': float(global_pos_rate),
        'interpretation': 'Maximum difference in positive class rates'
    }
    
    # ====================================================================
    # METRIC 5: COVARIATE SHIFT INDEX (Mean Feature Discrepancy)
    # ====================================================================
    print("\n5. COVARIATE SHIFT INDEX (Mean Feature Discrepancy)")
    print("   What is the feature distribution shift between hospitals?")
    
    csi_values = {}
    for i, h1 in enumerate(hospitals):
        for h2 in hospitals[i+1:]:
            X_h1 = hospital_data[h1]['X'][:min(500, hospital_data[h1]['n_samples'])]
            X_h2 = hospital_data[h2]['X'][:min(500, hospital_data[h2]['n_samples'])]
            
            # Mean Feature Discrepancy: L2 distance of feature means
            mfd = float(np.linalg.norm(X_h1.mean(axis=0) - X_h2.mean(axis=0)))
            csi_values[f'H{h1}_vs_H{h2}'] = mfd
            
            print(f"   Hospital {h1} vs {h2}: MFD = {mfd:.6f}")
    
    results['covariate_shift_index'] = {
        'pairwise': csi_values,
        'average': float(np.mean(list(csi_values.values()))),
        'metric_name': 'Mean Feature Discrepancy (MFD)',
        'formula': '||mean(X₁) - mean(X₂)||₂',
        'interpretation': 'Higher = greater feature distribution shift'
    }
    
    # ====================================================================
    # METRIC 6: WASSERSTEIN DISTANCE
    # ====================================================================
    print("\n6. WASSERSTEIN DISTANCE (Label Distribution)")
    print("   Earth Mover's Distance for label distributions")
    
    wd_values = {}
    for i, h1 in enumerate(hospitals):
        for h2 in hospitals[i+1:]:
            y_h1 = hospital_data[h1]['y']
            y_h2 = hospital_data[h2]['y']
            
            # Wasserstein between class distributions
            wd = float(wasserstein_distance(
                np.bincount(y_h1.astype(int), minlength=2) / len(y_h1),
                np.bincount(y_h2.astype(int), minlength=2) / len(y_h2)
            ))
            wd_values[f'H{h1}_vs_H{h2}'] = wd
            
            print(f"   Hospital {h1} vs {h2}: Wasserstein = {wd:.6f}")
    
    results['wasserstein_distance'] = {
        'pairwise': wd_values,
        'average': float(np.mean(list(wd_values.values()))),
        'interpretation': 'Range [0,1]: 0=identical, 1=different'
    }
    
    # ====================================================================
    # COMPOSITE NON-IID SCORE
    # ====================================================================
    # Weighted combination of all 6 metrics
    composite_score = (
        0.20 * results['label_heterogeneity']['average'] +
        0.15 * np.tanh(results['feature_heterogeneity']['average']) +
        0.10 * np.tanh(results['correlation_heterogeneity']['average'] / 10) +
        0.25 * results['class_imbalance_heterogeneity']['max_diff'] +
        0.15 * np.tanh(results['covariate_shift_index']['average']) +
        0.15 * results['wasserstein_distance']['average']
    )
    
    results['composite_non_iid_score'] = float(composite_score)
    
    # Severity classification
    if composite_score < 0.15:
        severity = 'MINIMAL'
        implications = 'Data is nearly IID. Standard FL should work well.'
        recommended_algo = 'FedAvg'
    elif composite_score < 0.30:
        severity = 'LOW'
        implications = 'Slight Non-IID. FedAvg acceptable, watch convergence.'
        recommended_algo = 'FedAvg'
    elif composite_score < 0.50:
        severity = 'MODERATE'
        implications = 'Confirmed Non-IID. Consider fairness-aware aggregation.'
        recommended_algo = 'FedAvg + fairness'
    elif composite_score < 0.70:
        severity = 'HIGH'
        implications = 'Severe Non-IID. Use FedProx or FedWeight, test fairness.'
        recommended_algo = 'FedProx or FedWeight'
    else:
        severity = 'CRITICAL'
        implications = 'Extreme heterogeneity. Need domain adaptation + balancing.'
        recommended_algo = 'Domain-adaptive FL + fairness'
    
    results['severity_assessment'] = {
        'level': severity,
        'implications': implications,
        'recommended_algorithm': recommended_algo
    }
    
    # ====================================================================
    # PRINT SUMMARY
    # ====================================================================
    print("\n" + "="*70)
    print("NON-IID QUANTIFICATION SUMMARY")
    print("="*70)
    
    print(f"\nComposite Non-IID Score: {composite_score:.4f}")
    print(f"Severity Level: {severity}")
    print(f"\nImplications: {implications}")
    print(f"Recommended Algorithm: {recommended_algo}")
    
    return results

def calculate_label_distribution_divergence(y, sens_test):
    """
    Measure label distribution differences across hospitals
    High difference = Higher Non-IID-ness
    """
    hospital_label_rates = {}
    
    print("\n" + "-"*60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("-"*60)
    
    for h_id in [1, 2, 3]:
        mask = sens_test['hospital_id'] == h_id
        y_h = y[mask]
        
        if len(y_h) == 0:
            continue
        
        pos_rate = float(y_h.mean())
        hospital_label_rates[h_id] = pos_rate
        
        print(f"Hospital {h_id}: {pos_rate*100:.2f}% positive")
    
    if len(hospital_label_rates) > 1:
        rates = list(hospital_label_rates.values())
        max_diff = max(rates) - min(rates)
        print(f"\nMax Label Distribution Difference: {max_diff*100:.2f}%")
        return max_diff
    
    return 0.0

def calculate_feature_distribution_divergence(X, sens_test):
    """
    Measure feature distribution differences using Jensen-Shannon divergence
    """
    print("\n" + "-"*60)
    print("FEATURE DISTRIBUTION ANALYSIS (Covariate Shift)")
    print("-"*60)
    
    hospitals = [1, 2, 3]
    hospital_data = {}
    
    for h_id in hospitals:
        mask = sens_test['hospital_id'] == h_id
        X_h = X[mask]
        if len(X_h) > 0:
            hospital_data[h_id] = X_h
    
    def compute_js_pairwise(X1, X2, n_features=None):
        """Compute average JS divergence across features"""
        if n_features is None:
            n_features = min(X1.shape[1], X2.shape[1])
        
        js_scores = []
        
        for feat_idx in range(n_features):
            f1 = X1[:, feat_idx]
            f2 = X2[:, feat_idx]
            
            # Normalize to [0, 1]
            mn = min(f1.min(), f2.min())
            mx = max(f1.max(), f2.max())
            
            if mx - mn < 1e-6:
                js_scores.append(0.0)
                continue
            
            f1_norm = (f1 - mn) / (mx - mn)
            f2_norm = (f2 - mn) / (mx - mn)
            
            # Create histograms
            bins = np.linspace(0, 1, 30)
            p, _ = np.histogram(f1_norm, bins=bins, density=True)
            q, _ = np.histogram(f2_norm, bins=bins, density=True)
            
            # Normalize to probability distributions
            p = p / (p.sum() + 1e-12)
            q = q / (q.sum() + 1e-12)
            
            # Jensen-Shannon divergence
            js = jensenshannon(p, q)
            js_scores.append(js)
        
        return float(np.mean(js_scores))
    
    # Pairwise comparisons
    pairs = [(1, 2), (1, 3), (2, 3)]
    js_values = {}
    
    print("\nJensen-Shannon Divergence (Feature Distribution Similarity):")
    
    for h1, h2 in pairs:
        if h1 in hospital_data and h2 in hospital_data:
            js = compute_js_pairwise(hospital_data[h1], hospital_data[h2], n_features=20)
            js_values[f"{h1}-{h2}"] = js
            print(f"  Hospital {h1} vs {h2}: {js:.4f}")
    
    if js_values:
        avg_js = float(np.mean(list(js_values.values())))
        print(f"\nAverage JS Divergence: {avg_js:.4f}")
        return avg_js, js_values
    
    return 0.0, {}

def calculate_feature_correlation_divergence(X, sens_test):
    """
    Measure correlation pattern differences across hospitals
    Different correlation structures = Higher Non-IID-ness
    """
    print("\n" + "-"*60)
    print("CORRELATION PATTERN ANALYSIS")
    print("-"*60)
    
    hospitals = [1, 2, 3]
    correlation_diffs = []
    
    for h_id in hospitals:
        mask = sens_test['hospital_id'] == h_id
        X_h = X[mask]
        
        if len(X_h) < 5:
            continue
        
        # Sample first 20 features for correlation analysis
        X_sample = X_h[:, :min(20, X_h.shape[1])]
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X_sample.T)
        
        # Get upper triangle as vector
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        upper_tri = np.nan_to_num(upper_tri, nan=0.0)
        
        print(f"Hospital {h_id}: Mean |correlation| = {np.abs(upper_tri).mean():.4f}")
    
    print(f"\n→ Correlation patterns help measure statistical diversity")

def analyze_demographic_distribution(sens_test):
    """
    Analyze demographic diversity within and across hospitals
    """
    print("\n" + "-"*60)
    print("DEMOGRAPHIC DISTRIBUTION ANALYSIS")
    print("-"*60)
    
    hospitals = [1, 2, 3]
    
    for h_id in hospitals:
        mask = sens_test['hospital_id'] == h_id
        sens_h = sens_test[mask]
        
        if len(sens_h) == 0:
            continue
        
        print(f"\nHospital {h_id}:")
        print(f"  Total Samples: {len(sens_h):,}")
        
        if 'gender' in sens_h.columns:
            print(f"  Gender Distribution:")
            gender_dist = sens_h['gender'].value_counts()
            for gender, count in gender_dist.items():
                print(f"    {gender}: {count} ({count/len(sens_h)*100:.1f}%)")
        
        if 'race' in sens_h.columns:
            print(f"  Race Distribution:")
            race_dist = sens_h['race'].value_counts()
            for race, count in race_dist.head(3).items():
                print(f"    {race}: {count} ({count/len(sens_h)*100:.1f}%)")

def analyze_non_iid():
    """Run complete Non-IID analysis with 6-metric framework"""
    print("\n" + "="*70)
    print("PHASE 3: NON-IID QUANTIFICATION ANALYSIS (6-METRIC FRAMEWORK)")
    print("="*70)
    
    # Load TRAINING data for proper Non-IID quantification
    X_train, y_train, sens_train, is_balanced = load_balanced_train_data()
    
    # Load feature names
    data_dir = Path('data/processed_data_balanced') if is_balanced \
               else Path('data/processed_data')
    feature_names_file = data_dir / 'feature_names_balanced.txt'
    
    if not feature_names_file.exists():
        feature_names_file = Path('data/processed_data/feature_names.txt')
    
    with open(feature_names_file, 'r') as f:
        feature_names = f.read().splitlines()
    
    print(f"\nDataset Type: {'BALANCED' if is_balanced else 'UNBALANCED'}")
    print(f"Training Samples: {len(X_train):,}")
    print(f"Features: {len(feature_names)}")
    
    # ====================================================================
    # Comprehensive 6-metric Non-IID analysis
    # ====================================================================
    results = comprehensive_non_iid_analysis(X_train, y_train, sens_train, feature_names)
    
    # ====================================================================
    # VISUALIZATION
    # ====================================================================
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Metrics Overview
    metric_names = [
        'Label Het.',
        'Feature Het.',
        'Correlation Het.',
        'Class Imbalance',
        'Mean Feature Disc.',
        'Wasserstein'
    ]
    metric_values = [
        results['label_heterogeneity']['average'],
        np.tanh(results['feature_heterogeneity']['average']),
        np.tanh(results['correlation_heterogeneity']['average'] / 10),
        results['class_imbalance_heterogeneity']['max_diff'],
        np.tanh(results['covariate_shift_index']['average']),
        results['wasserstein_distance']['average']
    ]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(metric_names)))
    axes[0, 0].bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Normalized Score')
    axes[0, 0].set_title('Individual Non-IID Metrics', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Composite Score
    composite = results['composite_non_iid_score']
    severity_colors = {
        'MINIMAL': '#2ecc71',
        'LOW': '#f39c12',
        'MODERATE': '#e67e22',
        'HIGH': '#e74c3c',
        'CRITICAL': '#c0392b'
    }
    color = severity_colors.get(results['severity_assessment']['level'], '#95a5a6')
    
    axes[0, 1].barh(['Non-IID Score'], [composite], color=color, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlim(0, 1.0)
    axes[0, 1].set_title(f"Composite Non-IID Score: {composite:.4f}\n({results['severity_assessment']['level']})",
                         fontweight='bold', fontsize=12)
    axes[0, 1].text(composite + 0.02, 0, f'{composite:.4f}', va='center', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Subplot 3: Label Distribution Per Hospital
    label_het_data = results['label_heterogeneity']['per_hospital']
    hospitals = list(label_het_data.keys())
    js_values = list(label_het_data.values())
    
    colors_labels = ['#FF6B6B' if v > 0.1 else '#4ECDC4' for v in js_values]
    axes[1, 0].bar([f'Hospital {h}' for h in hospitals], js_values,
                   color=colors_labels, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Jensen-Shannon Divergence')
    axes[1, 0].set_title('Label Heterogeneity Per Hospital', fontweight='bold', fontsize=12)
    axes[1, 0].axhline(y=0.1, color='red', linestyle='--', label='Threshold', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Summary Text
    axes[1, 1].axis('off')
    summary_text = f"""
COMPOSITE NON-IID SCORE: {composite:.4f}

Severity Level: {results['severity_assessment']['level']}

Implications:
{results['severity_assessment']['implications']}

Recommended Algorithm:
{results['severity_assessment']['recommended_algorithm']}

Individual Metrics:
• Label Heterogeneity: {results['label_heterogeneity']['average']:.4f}
• Feature Heterogeneity: {results['feature_heterogeneity']['average']:.4f}
• Correlation Heterogeneity: {results['correlation_heterogeneity']['average']:.4f}
• Class Imbalance: {results['class_imbalance_heterogeneity']['max_diff']:.4f}
• Mean Feature Discrepancy: {results['covariate_shift_index']['average']:.6f}
• Wasserstein Distance: {results['wasserstein_distance']['average']:.6f}
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/figures/04_non_iid_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to results/figures/04_non_iid_analysis_comprehensive.png")
    plt.close()
    
    # ====================================================================
    # SAVE RESULTS TO JSON
    # ====================================================================
    os.makedirs('results', exist_ok=True)
    
    # Convert all values to JSON-serializable format
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    results_serializable[key][k] = {
                        str(kk): float(vv) if isinstance(vv, (int, float, np.number)) else vv
                        for kk, vv in (v.items() if isinstance(v, dict) else enumerate(v))
                    }
                else:
                    results_serializable[key][k] = float(v) if isinstance(v, (int, float, np.number)) else v
        else:
            results_serializable[key] = float(value) if isinstance(value, (int, float, np.number)) else value
    
    results_serializable['dataset_type'] = 'BALANCED' if is_balanced else 'UNBALANCED'
    
    with open('results/non_iid_analysis_comprehensive.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("✓ Results saved to results/non_iid_analysis_comprehensive.json")
    
    print("\n" + "="*70)
    print("NON-IID ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    analyze_non_iid()