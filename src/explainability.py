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
    
    data_dir = Path('data/processed_data_balanced')
    
    if not (data_dir / 'X_test_balanced.npy').exists():
        if (Path('data/processed_data') / 'X_test_balanced.npy').exists():
            data_dir = Path('data/processed_data')
            is_balanced = True
        else:
            print("\n⚠️  Balanced data not found. Falling back to unbalanced data.")
            data_dir = Path('data/processed_data')
            is_balanced = False
    else:
        is_balanced = True
    
    if is_balanced:
        X_test = np.load(data_dir / 'X_test_balanced.npy').astype(np.float32)
        sens_test = pd.read_csv(data_dir / 'sensitive_attrs_test_balanced.csv')
        with open(data_dir / 'feature_names_balanced.txt', 'r') as f:
            feature_names = f.read().splitlines()
    else:
        X_test = np.load(data_dir / 'X_test.npy').astype(np.float32)
        sens_test = pd.read_csv(data_dir / 'sensitive_attrs_test.csv')
        with open(data_dir / 'feature_names.txt', 'r') as f:
            feature_names = f.read().splitlines()
    
    model_path = 'models/fedavg_global_model_best.keras'
    if not os.path.exists(model_path):
        model_path = 'models/fedavg_global_model_final.keras'
    if not os.path.exists(model_path):
        print(f" Error: Model not found at {model_path}")
        exit()
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f" ✓ Data loaded: X_test shape = {X_test.shape}")
    print(f" ✓ Features: {len(feature_names)} features")
    print(f" ✓ Model loaded successfully")
    
    return X_test, sens_test, feature_names, model, is_balanced

def safe_correlation(v1, v2):
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    
    valid_mask = ~np.isnan(v1) & ~np.isnan(v2)
    if np.sum(valid_mask) == 0: return 0.0
    
    v1m, v2m = v1[valid_mask], v2[valid_mask]
    if np.nanstd(v1m) < 1e-9 or np.nanstd(v2m) < 1e-9: return 0.0
    return float(np.corrcoef(v1m, v2m)[0, 1])

def validate_shap_against_clinical_literature(feature_names, importance):
    expected_high_importance = {
        'num_medications': 'Polypharmacy',
        'number_inpatient': 'Prior hospitalization',
        'number_emergency': 'ED visits',
        'time_in_hospital': 'Longer stay',
        'insulin': 'Insulin dependency',
        'age': 'Advanced age',
        'num_lab_procedures': 'More tests',
        'number_diagnoses': 'Multiple diagnoses',
        'diabetesMed': 'Diabetes medication intensity',
    }
    
    top_indices = np.argsort(importance)[-10:][::-1]
    top_features = set(feature_names[int(i)] for i in top_indices)
    
    expected_set = set(expected_high_importance.keys())
    matches = top_features & expected_set
    alignment_score = len(matches) / max(len(expected_set), 1)
    
    return {
        'alignment_score': float(alignment_score),
        'interpretation': ('CLINICALLY PLAUSIBLE' if alignment_score > 0.5 else 'INVESTIGATE'),
        'matched_expected': sorted(list(matches)),
        'unmatched_expected': sorted(list(expected_set - matches)),
        'top_10_features': list(top_features)
    }

def compute_shap_with_stability_assessment(X_test, sens_test, feature_names, model, n_repeats=3, sample_size=50):
    print("\nComputing SHAP values with stability assessment...")
    
    hospital_names = {1: 'Circulatory (Heart/Stroke)', 2: 'Metabolic (Diabetes/Kidney)', 3: 'Other (Respiratory)'}
    results = {}
    os.makedirs('results/figures', exist_ok=True)
    
    # Suppress KMeans memory leak warning on Windows
    os.environ["OMP_NUM_THREADS"] = "1"
    
    for h_id in [1, 2, 3]:
        mask = sens_test['hospital_id'] == h_id
        X_h = X_test[mask]
        
        if len(X_h) < 50: continue
        print(f"\n--- Hospital {h_id} ({hospital_names[h_id]}) ---")
        
        from sklearn.cluster import MiniBatchKMeans
        bg_size = min(100, len(X_h))
        kmeans = MiniBatchKMeans(n_clusters=bg_size, random_state=42, n_init=3)
        kmeans.fit(X_h)
        
        distances = np.linalg.norm(X_h[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :], axis=2)
        bg_indices = np.argmin(distances, axis=0)
        X_background = X_h[bg_indices]
        
        sample_size_actual = min(sample_size, len(X_h))
        sample_indices = np.random.choice(len(X_h), sample_size_actual, replace=False)
        X_sample = X_h[sample_indices]
        
        importance_runs = []
        print(f"Computing SHAP values ({n_repeats} runs)...")
        
        for run in range(n_repeats):
            print(f"  Run {run+1}/{n_repeats}...", end=' ', flush=True)
            np.random.seed(42 + run)
            explainer = shap.KernelExplainer(model.predict, X_background, link="logit")
            shap_values = explainer.shap_values(X_sample, nsamples=150)
            
            if isinstance(shap_values, list): shap_values = shap_values[0]
            
            importance = np.abs(shap_values).mean(axis=0)
            importance = np.nan_to_num(np.squeeze(importance))
            importance_runs.append(importance)
            print("✓")
            
        importance_final = np.mean(importance_runs, axis=0)
        importance_std = np.std(importance_runs, axis=0)
        importance_cv = importance_std / (np.abs(importance_final) + 1e-8)
        stability_score = 1.0 - np.mean(np.clip(importance_cv, 0, 1))
        
        print(f"  Stability Score: {stability_score:.3f}")
        validation = validate_shap_against_clinical_literature(feature_names, importance_final)
        
        top_indices_15 = np.argsort(importance_final)[-15:]
        top_features = [feature_names[int(i)] for i in top_indices_15]
        top_values = importance_final[top_indices_15]
        top_stds = importance_std[top_indices_15]
        
        plt.figure(figsize=(10, 7))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        plt.barh(top_features, top_values, xerr=top_stds, color=list(colors), alpha=0.7, capsize=5)
        plt.xlabel('Mean |SHAP Value| ± Std Dev')
        plt.title(f'Feature Importance (SHAP) - Hospital {h_id}\nStability: {stability_score:.3f}')
        plt.tight_layout()
        plt.savefig(f'results/figures/08_shap_importance_hospital_{h_id}.png', dpi=300)
        plt.close()
        
        top_indices = np.argsort(importance_final)[-10:][::-1]
        results[h_id] = {
            'hospital_name': hospital_names[h_id],
            'n_samples': int(len(X_h)),
            'importance_vector': importance_final.tolist(),
            'importance_std_vector': importance_std.tolist(),
            'stability_score': float(stability_score),
            'clinical_validation': validation,
            'top_10_features': [{'rank': r+1, 'feature': feature_names[i], 'importance': float(importance_final[i])} for r, i in enumerate(top_indices)]
        }
    return results

def run_shap_analysis():
    print("\n" + "="*70)
    print("PHASE 6: SHAP EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    X_test, sens_test, feature_names, model, is_balanced = load_resources()
    
    hospital_stats = compute_shap_with_stability_assessment(X_test, sens_test, feature_names, model)
    hospital_importance_vectors = {h_id: np.array(data['importance_vector']) for h_id, data in hospital_stats.items()}
    
    h1 = hospital_importance_vectors.get(1, np.zeros(len(feature_names)))
    h2 = hospital_importance_vectors.get(2, np.zeros(len(feature_names)))
    h3 = hospital_importance_vectors.get(3, np.zeros(len(feature_names)))
    
    corr_12 = safe_correlation(h1, h2)
    corr_13 = safe_correlation(h1, h3)
    corr_23 = safe_correlation(h2, h3)
    avg_consistency = (corr_12 + corr_13 + corr_23) / 3
    
    print(f"\nPearson Correlation (Cross-Hospital): {avg_consistency:.4f}")
    
    from scipy.spatial.distance import jensenshannon
    def compute_js(X1, X2):
        js = []
        for i in range(min(X1.shape[1], X2.shape[1])):
            f1, f2 = X1[:, i], X2[:, i]
            mn, mx = min(f1.min(), f2.min()), max(f1.max(), f2.max())
            if mn == mx: continue
            bins = np.linspace(mn, mx, 50)
            p, _ = np.histogram(f1, bins=bins, density=True)
            q, _ = np.histogram(f2, bins=bins, density=True)
            js.append(jensenshannon(p / (p.sum()+1e-12), q / (q.sum()+1e-12)))
        return np.mean(js)

    h1_d, h2_d, h3_d = X_test[sens_test['hospital_id']==1], X_test[sens_test['hospital_id']==2], X_test[sens_test['hospital_id']==3]
    avg_js = np.mean([compute_js(h1_d, h2_d), compute_js(h1_d, h3_d), compute_js(h2_d, h3_d)])
    
    results = {
        'dataset_type': 'BALANCED' if is_balanced else 'UNBALANCED',
        'consistency_analysis': {'corr_h1_h2': float(corr_12), 'average_consistency': float(avg_consistency)},
        'hospital_stats': hospital_stats
    }
    
    with open('results/shap_analysis.json', 'w') as f: json.dump(results, f, indent=2)
    print("\n✓ Analysis Complete. Saved to results/shap_analysis.json")

if __name__ == "__main__":
    run_shap_analysis()