import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import os
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1" # Fix KMeans Memory Warning

def load_resources():
    print("Loading resources for local explainability...")
    data_dir = Path('data/processed_data_balanced')
    
    if not (data_dir / 'X_test_balanced.npy').exists():
        data_dir = Path('data/processed_data')
        suffix = '_balanced' if (data_dir / 'X_test_balanced.npy').exists() else ''
    else:
        suffix = '_balanced'

    X_train = np.load(data_dir / f'X_train{suffix}.npy').astype(np.float32)
    y_train = np.load(data_dir / f'y_train{suffix}.npy').astype(np.float32)
    X_test = np.load(data_dir / f'X_test{suffix}.npy').astype(np.float32)
    y_test = np.load(data_dir / f'y_test{suffix}.npy').astype(np.float32)
    sens_train = pd.read_csv(data_dir / f'sensitive_attrs_train{suffix}.csv')
    sens_test = pd.read_csv(data_dir / f'sensitive_attrs_test{suffix}.csv')
    
    with open(data_dir / f'feature_names{suffix}.txt', 'r') as f:
        feature_names = f.read().splitlines()
        
    model_path = 'models/fedavg_global_model_best.keras'
    model = tf.keras.models.load_model(model_path, compile=False)
    
    return X_train, y_train, X_test, y_test, sens_train, sens_test, feature_names, model

def compute_local_hospital_explanations(X_test, y_test, sens_test, feature_names, model):
    print("\n" + "="*70)
    print("LOCAL HOSPITAL EXPLANATIONS")
    print("="*70)
    
    hospital_names = {1: 'Circulatory', 2: 'Metabolic', 3: 'Other'}
    local_results, hospital_importance_vectors = {}, {}
    
    for h_id in [1, 2, 3]:
        mask = sens_test['hospital_id'] == h_id
        X_h, y_h = X_test[mask], y_test[mask]
        
        if len(X_h) < 50: continue
        print(f"\n--- Hospital {h_id} ({hospital_names[h_id]}) ---")
        
        sample_size = min(50, len(X_h))
        X_sample = X_h[np.random.choice(len(X_h), sample_size, replace=False)]
        background = X_h[np.random.choice(len(X_h), min(100, len(X_h)), replace=False)]
        
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        if isinstance(shap_values, list): shap_values = shap_values[0]

        importance = np.nan_to_num(np.squeeze(np.abs(shap_values).mean(axis=0)))
        hospital_importance_vectors[h_id] = importance
        
        top_indices = np.argsort(importance)[-10:][::-1]
        
        # FIX: Ensure it saves as a list of dicts/tuples correctly for plotting later
        top_features = [{'feature': feature_names[i], 'importance': float(importance[i])} for i in top_indices]
        
        local_results[h_id] = {
            'hospital_name': hospital_names[h_id],
            'n_samples': int(len(X_h)),
            'top_10_features': top_features,
            'importance_vector': importance.tolist()
        }
    return local_results, hospital_importance_vectors

def compute_fairness_explanations(X_test, y_test, sens_test, feature_names, model):
    print("\n" + "="*70)
    print("FAIRNESS EXPLANATION")
    print("="*70)
    
    fairness_results = {'gender': {}}
    for gender in ['Female', 'Male']:
        mask = sens_test['gender'] == gender
        X_g = X_test[mask]
        
        if len(X_g) < 50: continue
        
        X_sample = X_g[np.random.choice(len(X_g), min(30, len(X_g)), replace=False)]
        background = X_g[np.random.choice(len(X_g), min(80, len(X_g)), replace=False)]
        
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample, nsamples=80)
        if isinstance(shap_values, list): shap_values = shap_values[0]

        importance = np.nan_to_num(np.squeeze(np.abs(shap_values).mean(axis=0)))
        top_indices = np.argsort(importance)[-5:][::-1]
        
        fairness_results['gender'][gender] = {
            'n_samples': int(len(X_g)),
            'top_5_features': [{'feature': feature_names[i], 'importance': float(importance[i])} for i in top_indices]
        }
    return fairness_results

def create_visualizations(local_results, fairness_results):
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (h_id, results) in enumerate(local_results.items()):
        features = [f['feature'] for f in results['top_10_features'][::-1]]
        importances = [f['importance'] for f in results['top_10_features'][::-1]]
        axes[idx].barh(features, importances, color='steelblue')
        axes[idx].set_title(f"Hospital {h_id}", fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/07a_local_hospital_importance.png', dpi=300)
    plt.close()
    
    if 'gender' in fairness_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for idx, (gender, res) in enumerate(fairness_results['gender'].items()):
            features = [f['feature'] for f in res['top_5_features'][::-1]]
            importances = [f['importance'] for f in res['top_5_features'][::-1]]
            axes[idx].barh(features, importances, color='#FF6B6B' if gender == 'Female' else '#4ECDC4')
            axes[idx].set_title(f"{gender}", fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/figures/07c_fairness_gender_importance.png', dpi=300)
        plt.close()

def run_local_explainability():
    X_train, y_train, X_test, y_test, sens_train, sens_test, feature_names, model = load_resources()
    local_results, hospital_importance_vectors = compute_local_hospital_explanations(X_test, y_test, sens_test, feature_names, model)
    fairness_results = compute_fairness_explanations(X_test, y_test, sens_test, feature_names, model)
    create_visualizations(local_results, fairness_results)
    
    os.makedirs('results', exist_ok=True)
    with open('results/local_hospital_explanations.json', 'w') as f: json.dump(local_results, f, indent=2)
    print("\n✓ Process Complete. Results and Figures saved.")

if __name__ == "__main__":
    run_local_explainability()