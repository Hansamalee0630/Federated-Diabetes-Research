
import tensorflow as tf
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
import json
from scipy.stats import pearsonr

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_resources():
    """Load balanced data and trained model"""
    print("Loading resources...")
    
    # CRITICAL: Use balanced data path
    data_dir = Path('data/processed_data_balanced')
    
    if not (data_dir / 'X_test_balanced.npy').exists():
        print("\n⚠️  Balanced data not found. Falling back to unbalanced data.")
        data_dir = Path('data/processed_data')
        is_balanced = False
    else:
        is_balanced = True
    
    X_test = np.load(data_dir / 'X_test_balanced.npy').astype(np.float32) \
             if is_balanced else np.load(data_dir / 'X_test.npy').astype(np.float32)
    
    sens_test_file = 'sensitive_attrs_test_balanced.csv' if is_balanced \
                     else 'sensitive_attrs_test.csv'
    sens_test = pd.read_csv(data_dir / sens_test_file)
    
    with open(data_dir / 'feature_names_balanced.txt', 'r') as f:
        feature_names = f.read().splitlines()
    
    model_path = 'models/fedavg_global_model_best.keras'
    if not os.path.exists(model_path):
        print(f" Error: Model not found at {model_path}")
        exit()
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f" ✓ Data loaded: X_test shape = {X_test.shape}")
    print(f" ✓ Features: {len(feature_names)} features")
    print(f" ✓ Model loaded successfully")
    
    return X_test, sens_test, feature_names, model, is_balanced

def safe_correlation(v1, v2):
    """
    Calculate Pearson correlation safely, handling zero-variance cases
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    
    # Mask invalid values
    valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
    if np.sum(valid_mask) == 0:
        return 0.0
    
    v1m = v1[valid_mask]
    v2m = v2[valid_mask]
    
    # Check for zero variance
    if np.nanstd(v1m) < 1e-9 or np.nanstd(v2m) < 1e-9:
        return 0.0
    
    return float(np.corrcoef(v1m, v2m)[0, 1])

def validate_shap_against_clinical_literature(feature_names, importance):
    """
    Cross-check SHAP results against known readmission risk factors
    
    Source: Donzé et al. (2016), BMJ, "Readmission Risk Factors"
    and Jencks et al. (2013), NEJM, "30-day Mortality & Readmission"
    """
    
    # Established readmission risk factors (diabetes-specific)
    expected_high_importance = {
        'num_medications': 'Polypharmacy → drug interactions, confusion',
        'number_inpatient': 'Prior hospitalization → severity marker',
        'number_emergency': 'ED visits → unstable status',
        'time_in_hospital': 'Longer stay → discharge too early',
        'insulin': 'Insulin dependency → brittle control',
        'age': 'Advanced age → comorbidities, fragility',
        'num_lab_procedures': 'More tests → disease complexity',
        'number_diagnoses': 'Multiple diagnoses → comorbidities',
        'diabetesMed': 'Diabetes medication intensity',
    }
    
    # Get top 10 features from model
    top_indices = np.argsort(importance)[-10:][::-1]
    top_features = set(feature_names[int(i)] for i in top_indices)
    
    # Check overlap
    expected_set = set(expected_high_importance.keys())
    matches = top_features & expected_set
    
    alignment_score = len(matches) / max(len(expected_set), 1)
    
    return {
        'alignment_score': alignment_score,
        'interpretation': (
            'CLINICALLY PLAUSIBLE' if alignment_score > 0.6 else 
            'INVESTIGATE' if alignment_score > 0.3 else
            'ANOMALOUS - Check data'
        ),
        'matched_expected': sorted(list(matches)),
        'unmatched_expected': sorted(list(expected_set - matches)),
        'top_10_features': list(top_features),
        'explanation': (
            'Model emphasizes established risk factors' if alignment_score > 0.6 else
            'Model may have learned spurious patterns - verify with domain experts'
        )
    }


def compute_shap_with_stability_assessment(X_test, sens_test, feature_names, 
                                          model, n_repeats=3, sample_size=50):
    """
    Compute SHAP values with reproducibility assessment
    
    Parameters:
    -----------
    n_repeats : int
        Number of independent SHAP runs (recommended: 3+)
    sample_size : int
        Patients per hospital for SHAP (recommended: 50+)
    
    Returns:
    --------
    dict with hospital-level SHAP analysis including stability scores
    """
    
    print("\nComputing SHAP values with stability assessment...")
    
    hospital_names = {
        1: 'Circulatory (Heart/Stroke)',
        2: 'Metabolic (Diabetes/Kidney)',
        3: 'Other (Respiratory/Digestive)'
    }
    
    results = {}
    os.makedirs('results/figures', exist_ok=True)
    
    for h_id in [1, 2, 3]:
        mask = sens_test['hospital_id'] == h_id
        X_h = X_test[mask]
        
        if len(X_h) < 50:
            print(f"\nHospital {h_id}: Skipping (only {len(X_h)} samples)")
            continue
        
        print(f"\n--- Hospital {h_id} ({hospital_names[h_id]}) ---")
        print(f"Total samples: {len(X_h):,}")
        
        # ====================================================================
        # STEP 1: DIVERSITY-AWARE BACKGROUND SELECTION
        # ====================================================================
        # Use clustering to get representative samples
        from sklearn.cluster import MiniBatchKMeans
        
        bg_size = min(100, len(X_h))
        print(f"Selecting {bg_size} diverse background samples...")
        
        kmeans = MiniBatchKMeans(n_clusters=bg_size, random_state=42, n_init=3)
        kmeans.fit(X_h)
        
        # Find closest sample to each cluster center
        distances = np.linalg.norm(X_h[:, np.newaxis, :] - 
                                  kmeans.cluster_centers_[np.newaxis, :, :], 
                                  axis=2)
        bg_indices = np.argmin(distances, axis=0)
        X_background = X_h[bg_indices]
        
        # ====================================================================
        # STEP 2: SAMPLE SELECTION
        # ====================================================================
        sample_size_actual = min(sample_size, len(X_h))
        sample_indices = np.random.choice(len(X_h), sample_size_actual, replace=False)
        X_sample = X_h[sample_indices]
        
        print(f"Sample size: {sample_size_actual}")
        
        # ====================================================================
        # STEP 3: COMPUTE SHAP WITH MULTIPLE RUNS
        # ====================================================================
        importance_runs = []
        
        print(f"Computing SHAP values ({n_repeats} runs for stability)...")
        
        for run in range(n_repeats):
            print(f"  Run {run+1}/{n_repeats}...", end=' ', flush=True)
            
            np.random.seed(42 + run)
            
            explainer = shap.KernelExplainer(
                model.predict,
                X_background,
                link="logit"  # Binary classification
            )
            
            # CRITICAL: nsamples must be high for medical decisions
            shap_values = explainer.shap_values(X_sample, nsamples=200)
            
            # Handle binary classification (returns list)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Positive class
            
            # Compute feature importance (mean absolute SHAP) and ensure 1-D
            importance = np.abs(shap_values).mean(axis=0)
            importance = np.squeeze(importance)
            importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            
            importance_runs.append(importance)
            print("✓")
        
        # ====================================================================
        # STEP 4: STABILITY ASSESSMENT
        # ====================================================================
        importance_final = np.mean(importance_runs, axis=0)
        importance_std = np.std(importance_runs, axis=0)
        
        # Coefficient of variation per feature
        importance_cv = importance_std / (np.abs(importance_final) + 1e-8)
        
        # Stability score: how consistent are rankings?
        stability_score = 1.0 - np.mean(np.clip(importance_cv, 0, 1))
        
        print(f"Stability Assessment:")
        print(f"  Mean CV: {importance_cv.mean():.3f}")
        print(f"  Stability Score: {stability_score:.3f}")
        
        if stability_score > 0.8:
            print(f"  ✓ RELIABLE (clinically actionable)")
        elif stability_score > 0.6:
            print(f"  ~ MODERATE (use with caution)")
        else:
            print(f"  ⚠ UNRELIABLE (need more samples)")
        
        # ====================================================================
        # STEP 5: CLINICAL VALIDATION
        # ====================================================================
        validation = validate_shap_against_clinical_literature(
            feature_names, importance_final
        )
        
        print(f"Clinical Validation:")
        print(f"  Alignment with literature: {validation['alignment_score']*100:.0f}%")
        print(f"  Status: {validation['interpretation']}")
        
        # ====================================================================
        # STEP 6: VISUALIZATION
        # ====================================================================
        top_indices_15 = np.argsort(importance_final)[-15:][::-1]
        top_features = [feature_names[int(i)] for i in top_indices_15]
        top_values = importance_final[top_indices_15]
        top_stds = importance_std[top_indices_15]
        
        plt.figure(figsize=(10, 7))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        plt.barh(top_features, top_values, xerr=top_stds, color=list(colors),
                alpha=0.7, capsize=5)
        plt.xlabel('Mean |SHAP Value| ± Std Dev')
        plt.title(f'Feature Importance (SHAP) - Hospital {h_id}\n({hospital_names[h_id]})\nStability: {stability_score:.3f}', 
                  fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'results/figures/08_shap_importance_hospital_{h_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ====================================================================
        # STEP 7: SAVE RESULTS
        # ====================================================================
        top_indices = np.argsort(importance_final)[-10:][::-1]
        top_features_list = [
            {
                'rank': rank + 1,
                'feature': feature_names[int(i)],
                'importance': float(importance_final[int(i)]),
                'std': float(importance_std[int(i)])
            }
            for rank, i in enumerate(top_indices)
        ]
        
        results[h_id] = {
            'hospital_name': hospital_names[h_id],
            'n_samples': int(len(X_h)),
            'n_shap_samples': int(sample_size_actual),
            'importance_vector': importance_final.tolist(),
            'importance_std_vector': importance_std.tolist(),
            'stability_score': float(stability_score),
            'clinical_validation': validation,
            'top_10_features': top_features_list
        }
    
    return results


def run_shap_analysis():
    """Run per-hospital SHAP analysis"""
    print("\n" + "="*70)
    print("PHASE 6: SHAP EXPLAINABILITY ANALYSIS (WITH STABILITY ASSESSMENT)")
    print("="*70)
    
    X_test, sens_test, feature_names, model, is_balanced = load_resources()
    
    print(f"\nDataset Type: {'BALANCED' if is_balanced else 'UNBALANCED'}")
    print(f"Test Samples: {len(X_test):,}")
    
    # Compute SHAP with stability assessment
    hospital_importance_vectors = {}
    hospital_stats = compute_shap_with_stability_assessment(
        X_test, sens_test, feature_names, model,
        n_repeats=3,
        sample_size=50
    )
    
    # Extract importance vectors for consistency analysis
    for h_id, h_data in hospital_stats.items():
        hospital_importance_vectors[h_id] = np.array(h_data['importance_vector'])
    
    # ====================================================================
    # STEP 3: Cross-Hospital Consistency Analysis
    # ====================================================================
    print("\n" + "="*70)
    print("CROSS-HOSPITAL CONSISTENCY ANALYSIS")
    print("="*70)
    
    
    # ====================================================================
    # STEP 2: Cross-Hospital Consistency Analysis
    # ====================================================================
    print("\n" + "="*70)
    print("CROSS-HOSPITAL CONSISTENCY ANALYSIS")
    print("="*70)
    
    h1 = hospital_importance_vectors.get(1, np.zeros(len(feature_names)))
    h2 = hospital_importance_vectors.get(2, np.zeros(len(feature_names)))
    h3 = hospital_importance_vectors.get(3, np.zeros(len(feature_names)))
    
    corr_12 = safe_correlation(h1, h2)
    corr_13 = safe_correlation(h1, h3)
    corr_23 = safe_correlation(h2, h3)
    
    avg_consistency = (corr_12 + corr_13 + corr_23) / 3
    
    print(f"\nPearson Correlation (Feature Importance Rankings):")
    print(f"  Hospital 1 vs Hospital 2: {corr_12:.4f}")
    print(f"  Hospital 1 vs Hospital 3: {corr_13:.4f}")
    print(f"  Hospital 2 vs Hospital 3: {corr_23:.4f}")
    print(f"  Average Consistency:      {avg_consistency:.4f}")
    
    if avg_consistency > 0.85:
        consistency_interpretation = "✓ HIGH CONSISTENCY"
        interpretation_detail = "Model learns similar patterns across hospitals"
    elif avg_consistency > 0.70:
        consistency_interpretation = "~ MODERATE CONSISTENCY"
        interpretation_detail = "Some divergence due to non-IID data, but explainability is reliable"
    else:
        consistency_interpretation = "⚠ LOW CONSISTENCY"
        interpretation_detail = "Non-IID data significantly affects model logic across hospitals"
    
    print(f"\n{consistency_interpretation}: {interpretation_detail}")
    
    # ====================================================================
    # STEP 3: Non-IID Impact on SHAP
    # ====================================================================
    print("\n" + "="*70)
    print("NON-IID IMPACT ANALYSIS")
    print("="*70)
    
    from scipy.spatial.distance import jensenshannon
    
    def compute_js_divergence(X_h1, X_h2):
        """Jensen-Shannon divergence between feature distributions"""
        js_scores = []
        for i in range(min(X_h1.shape[1], X_h2.shape[1])):
            f1 = X_h1[:, i]
            f2 = X_h2[:, i]
            
            # Normalize to probabilities
            mn, mx = min(f1.min(), f2.min()), max(f1.max(), f2.max())
            if mn == mx:
                js_scores.append(0.0)
                continue
            
            bins = np.linspace(mn, mx, 50)
            p, _ = np.histogram(f1, bins=bins, density=True)
            q, _ = np.histogram(f2, bins=bins, density=True)
            
            p = p / (p.sum() + 1e-12)
            q = q / (q.sum() + 1e-12)
            
            js_scores.append(jensenshannon(p, q))
        
        return np.mean(js_scores)
    
    print("\nData Distribution Divergence (Jensen-Shannon):")
    
    h1_data = X_test[sens_test['hospital_id'] == 1]
    h2_data = X_test[sens_test['hospital_id'] == 2]
    h3_data = X_test[sens_test['hospital_id'] == 3]
    
    pairs = [(1, 2, h1_data, h2_data), 
             (1, 3, h1_data, h3_data), 
             (2, 3, h2_data, h3_data)]
    
    js_scores = []
    for h_a, h_b, X_a, X_b in pairs:
        if len(X_a) > 0 and len(X_b) > 0:
            js = compute_js_divergence(X_a, X_b)
            js_scores.append(js)
            print(f"  Hospital {h_a} vs {h_b}: JS = {js:.4f}")
    
    if js_scores:
        avg_js = np.mean(js_scores)
        print(f"  Average JS Divergence: {avg_js:.4f}")
        
        if avg_js > 0.1:
            print(f"  → High data heterogeneity (Non-IID)")
        else:
            print(f"  → Low data heterogeneity (IID-like)")
    
    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    results = {
        'dataset_type': 'BALANCED' if is_balanced else 'UNBALANCED',
        'consistency_analysis': {
            'corr_h1_h2': float(corr_12),
            'corr_h1_h3': float(corr_13),
            'corr_h2_h3': float(corr_23),
            'average_consistency': float(avg_consistency),
            'interpretation': consistency_interpretation
        },
        'hospital_stats': {k: v for k, v in hospital_stats.items() 
                          if isinstance(v, dict) and 'stability_score' in v}
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/shap_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results/shap_analysis.json")
    print("✓ Figures saved to results/figures/")
    
    print("\n" + "="*70)
    print("SHAP ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_shap_analysis()