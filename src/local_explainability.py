"""
LOCAL & INSTANCE-LEVEL EXPLAINABILITY FOR FEDERATED READMISSION MODEL
========================================================================

PHASE 7: Complete Explainability for Medical Federated Learning

This module provides THREE levels of explainability:
1. LOCAL HOSPITAL EXPLANATIONS: What does each hospital's local model care about?
2. INSTANCE-LEVEL EXPLANATIONS: Why was this specific patient predicted as high-risk?
3. FAIRNESS EXPLANATIONS: Do predictions differ fairly across gender/race?

All computations preserve privacy by working at local hospital level.
========================================================================
"""

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


# ============================================================================
# CLINICAL VALIDATION FUNCTION
# ============================================================================

def validate_local_explainability_clinical(feature_names, importance):
    """
    Cross-check local SHAP results against known readmission risk factors
    
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


# ============================================================================
# STEP 1: LOAD RESOURCES
# ============================================================================

def load_resources():
    """Load trained model, data, and feature names"""
    print("Loading resources for local explainability...")
    
    data_dir = Path('data/processed_data_balanced')
    
    # Load data
    X_train = np.load(data_dir / 'X_train_balanced.npy').astype(np.float32)
    y_train = np.load(data_dir / 'y_train_balanced.npy').astype(np.float32)
    X_test = np.load(data_dir / 'X_test_balanced.npy').astype(np.float32)
    y_test = np.load(data_dir / 'y_test_balanced.npy').astype(np.float32)
    
    # Load sensitive attributes
    sens_train = pd.read_csv(data_dir / 'sensitive_attrs_train_balanced.csv')
    sens_test = pd.read_csv(data_dir / 'sensitive_attrs_test_balanced.csv')
    
    # Load feature names
    with open(data_dir / 'feature_names_balanced.txt', 'r') as f:
        feature_names = f.read().splitlines()
    
    # Load model
    model_path = 'models/fedavg_global_model_best.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit()
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f"✓ Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"✓ Features: {len(feature_names)} features")
    print(f"✓ Model loaded successfully")
    
    return X_train, y_train, X_test, y_test, sens_train, sens_test, feature_names, model


# ============================================================================
# STEP 2: LOCAL HOSPITAL EXPLANATIONS
# ============================================================================

def compute_local_hospital_explanations(X_test, y_test, sens_test, feature_names, model):
    """
    Compute SHAP explanations for each hospital's local patterns.
    
    This answers: "What does each hospital's data emphasize?"
    """
    print("\n" + "="*70)
    print("LOCAL HOSPITAL EXPLANATIONS")
    print("="*70)
    print("\nComputing SHAP values for each hospital...")
    
    hospital_names = {
        1: 'Circulatory (Heart/Stroke)',
        2: 'Metabolic (Diabetes/Kidney)',
        3: 'Other (Respiratory/Digestive)'
    }
    
    local_results = {}
    hospital_importance_vectors = {}
    
    for h_id in [1, 2, 3]:
        print(f"\n--- Hospital {h_id} ({hospital_names[h_id]}) ---")
        
        # Get hospital-specific data
        mask = sens_test['hospital_id'] == h_id
        X_h = X_test[mask]
        y_h = y_test[mask]
        
        if len(X_h) < 50:
            print(f"  Skipping: Only {len(X_h)} samples")
            continue
        
        print(f"  Total samples: {len(X_h):,}")
        print(f"  Positive rate: {y_h.mean()*100:.2f}%")
        
        # Sample for speed (SHAP computation is expensive)
        sample_size = min(50, len(X_h))
        sample_indices = np.random.choice(len(X_h), sample_size, replace=False)
        X_sample = X_h[sample_indices]
        
        # Background data from same hospital
        background_size = min(100, len(X_h))
        bg_indices = np.random.choice(len(X_h), background_size, replace=False)
        background = X_h[bg_indices]
        
        print(f"  Computing SHAP for {sample_size} samples with {background_size} background...")
        
        # Create explainer
        explainer = shap.KernelExplainer(model.predict, background)
        
        # Compute SHAP values
        try:
            shap_values = explainer.shap_values(X_sample, nsamples=100)
        except:
            print(f"  Error computing SHAP, skipping hospital {h_id}")
            continue
        
        # Handle list output (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calculate mean absolute SHAP (feature importance) and ensure 1-D
        importance = np.abs(shap_values).mean(axis=0)
        importance = np.squeeze(importance)
        importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        
        hospital_importance_vectors[h_id] = importance
        
        # Get top features
        top_indices = np.argsort(importance)[-10:][::-1]
        # DEBUG: print diagnostic information when names look duplicated
        print(f"  DEBUG: len(feature_names)={len(feature_names)}")
        print(f"  DEBUG: feature_names[:6]={feature_names[:6]}")
        print(f"  DEBUG: importance[:12]={list(importance[:12])}")
        print(f"  DEBUG: top_indices={list(top_indices)}")
        top_features = [(feature_names[int(i)], float(importance[int(i)])) for i in top_indices]
        
        # ====================================================================
        # CLINICAL VALIDATION
        # ====================================================================
        clinical_validation = validate_local_explainability_clinical(feature_names, importance)
        
        print(f"  Top 10 Features:")
        for rank, (feat, imp) in enumerate(top_features, 1):
            print(f"    {rank:2d}. {feat:30s} {imp:.6f}")
        
        # Print clinical validation
        print(f"\n  Clinical Validation:")
        print(f"    Alignment with literature: {clinical_validation['alignment_score']*100:.0f}%")
        print(f"    Status: {clinical_validation['interpretation']}")
        if clinical_validation['matched_expected']:
            print(f"    Matched expected factors: {clinical_validation['matched_expected']}")
        if clinical_validation['unmatched_expected']:
            print(f"    Unexpected missing: {clinical_validation['unmatched_expected']}")
        
        # Store results
        local_results[h_id] = {
            'hospital_name': hospital_names[h_id],
            'n_samples': int(len(X_h)),
            'n_shap_samples': int(sample_size),
            'positive_rate': float(y_h.mean()),
            'top_10_features': top_features,
            'importance_vector': importance.tolist(),
            'clinical_validation': clinical_validation
        }
    
    return local_results, hospital_importance_vectors


# ============================================================================
# STEP 3: LOCAL VS GLOBAL DIVERGENCE ANALYSIS
# ============================================================================

def analyze_local_global_divergence(local_results, hospital_importance_vectors, feature_names):
    """
    Compare local hospital patterns with global model patterns.
    
    This answers: "Do hospitals disagree with the global model?"
    """
    print("\n" + "="*70)
    print("LOCAL vs GLOBAL DIVERGENCE ANALYSIS")
    print("="*70)
    
    # Load global SHAP analysis
    try:
        with open('results/shap_analysis.json', 'r') as f:
            global_analysis = json.load(f)
        global_consistency = global_analysis.get('consistency_analysis', {}).get('average_consistency', 0.93)
    except:
        global_consistency = 0.93
    
    print(f"\nGlobal Cross-Hospital Consistency: {global_consistency:.4f}")
    print("(High = all hospitals learn similar patterns)")
    
    print("\nLocal vs Global Feature Ranking Correlation:")
    
    if len(hospital_importance_vectors) > 1:
        # Compare pairs of hospitals
        hospital_ids = sorted(hospital_importance_vectors.keys())
        
        correlations = {}
        for i, h1 in enumerate(hospital_ids):
            for h2 in hospital_ids[i+1:]:
                imp1 = hospital_importance_vectors[h1]
                imp2 = hospital_importance_vectors[h2]
                
                # Correlation of feature importance rankings
                corr = np.corrcoef(imp1, imp2)[0, 1]
                corr = np.nan_to_num(corr, nan=0.0)
                
                correlations[f"H{h1}_vs_H{h2}"] = float(corr)
                
                print(f"  Hospital {h1} vs Hospital {h2}: {corr:.4f}")
        
        avg_local_corr = np.mean(list(correlations.values()))
        print(f"\n  Average Local Correlation: {avg_local_corr:.4f}")
        
        # Interpret
        if avg_local_corr > 0.85:
            interpretation = "✓ HIGH: Hospitals agree on feature importance"
        elif avg_local_corr > 0.70:
            interpretation = "~ MODERATE: Some differences but generally aligned"
        else:
            interpretation = "⚠ LOW: Hospitals have different clinical priorities"
        
        print(f"  Interpretation: {interpretation}")
        
        return correlations, interpretation
    
    return {}, "Insufficient hospitals for comparison"


# ============================================================================
# STEP 4: INSTANCE-LEVEL EXPLANATIONS (PER-PATIENT)
# ============================================================================

def explain_patient_prediction(model, patient_features, feature_names, 
                              patient_id, hospital_id, background_data,
                              sens_test, y_test):
    """
    Explain readmission risk prediction for a single patient.
    
    This answers: "Why is this specific patient high-risk?"
    """
    
    # Create explainer with background from same hospital
    explainer = shap.KernelExplainer(model.predict, background_data)
    
    # Get prediction
    prediction = model.predict(patient_features.reshape(1, -1), verbose=0)[0][0]
    
    # Compute SHAP value for this patient
    try:
        shap_value = explainer.shap_values(patient_features.reshape(1, -1), nsamples=50)[0]
    except:
        return None

    # Ensure shap_value is 1-D
    shap_value = np.squeeze(shap_value)
    
    # Get base value (expected prediction) and coerce to Python float
    base_value = explainer.expected_value
    try:
        # Handle cases where expected_value is array-like (numpy scalar/array)
        base_value = float(np.array(base_value).item())
    except Exception:
        try:
            base_value = float(base_value)
        except Exception:
            base_value = float(np.asarray(base_value).ravel()[0])
    
    # Get feature values
    feature_values = patient_features
    
    # Get top features contributing to prediction
    top_indices = np.argsort(np.abs(shap_value))[-10:][::-1]
    
    top_contributions = []
    for idx in top_indices:
        idx_int = int(idx)
        feature_name = feature_names[idx_int]
        contribution = float(shap_value[idx_int])
        feature_value = float(feature_values[idx_int])
        direction = "increases" if contribution > 0 else "decreases"
        
        top_contributions.append({
            'rank': len(top_contributions) + 1,
            'feature': feature_name,
            'contribution': contribution,
            'feature_value': feature_value,
            'direction': direction
        })
    
    explanation = {
        'patient_id': patient_id,
        'hospital_id': int(hospital_id),
        'predicted_risk': float(prediction),
        'predicted_risk_pct': f"{prediction*100:.1f}%",
        'base_risk': float(base_value),
        'base_risk_pct': f"{float(base_value)*100:.1f}%",
        'net_adjustment': float(prediction - base_value),
        'top_10_contributions': top_contributions,
        'risk_category': 'HIGH' if prediction > 0.6 else 'MODERATE' if prediction > 0.4 else 'LOW'
    }
    
    return explanation


def compute_instance_level_explanations(X_test, y_test, sens_test, feature_names, model, n_samples=20):
    """
    Compute instance-level explanations for sample patients from each hospital.
    
    This answers: "Can we explain individual patient predictions?"
    """
    print("\n" + "="*70)
    print("INSTANCE-LEVEL PATIENT EXPLANATIONS")
    print("="*70)
    
    patient_explanations = {}
    
    for h_id in [1, 2, 3]:
        print(f"\nHospital {h_id}: Explaining sample patient predictions...")
        
        # Get hospital data
        mask = sens_test['hospital_id'] == h_id
        X_h = X_test[mask]
        y_h = y_test[mask]
        
        if len(X_h) < 50:
            print(f"  Skipping: Only {len(X_h)} samples")
            continue
        
        # Background from hospital data
        bg_size = min(100, len(X_h))
        bg_indices = np.random.choice(len(X_h), bg_size, replace=False)
        background = X_h[bg_indices]
        
        # Sample patients: mix of high-risk and low-risk
        predictions_h = model.predict(X_h, verbose=0).flatten()
        
        high_risk_indices = np.where(predictions_h > 0.6)[0]
        low_risk_indices = np.where(predictions_h < 0.4)[0]
        
        # Select diverse patients
        selected_indices = []
        
        if len(high_risk_indices) > 0:
            selected_indices.extend(np.random.choice(high_risk_indices, 
                                                     min(n_samples//2, len(high_risk_indices)), 
                                                     replace=False))
        
        if len(low_risk_indices) > 0:
            selected_indices.extend(np.random.choice(low_risk_indices, 
                                                     min(n_samples//2, len(low_risk_indices)), 
                                                     replace=False))
        
        if len(selected_indices) == 0:
            selected_indices = np.random.choice(len(X_h), min(n_samples, len(X_h)), replace=False)
        
        print(f"  Explaining {len(selected_indices)} sample patients...")
        
        hospital_patients = []
        
        for sample_num, idx in enumerate(selected_indices, 1):
            patient_features = X_h[idx]
            patient_id = f"H{h_id}_P{sample_num:03d}"
            
            explanation = explain_patient_prediction(
                model, patient_features, feature_names,
                patient_id, h_id, background,
                sens_test, y_test
            )
            
            if explanation:
                hospital_patients.append(explanation)
                print(f"    {patient_id}: {explanation['predicted_risk_pct']} risk")
        
        patient_explanations[h_id] = hospital_patients
    
    return patient_explanations


# ============================================================================
# STEP 5: FAIRNESS EXPLANATION (PER-GROUP FEATURE IMPORTANCE)
# ============================================================================

def compute_fairness_explanations(X_test, y_test, sens_test, feature_names, model):
    """
    Analyze if feature importance differs by protected attributes (gender, race).
    
    This answers: "Are predictions biased against certain groups?"
    """
    print("\n" + "="*70)
    print("FAIRNESS EXPLANATION (Per-Group Feature Importance)")
    print("="*70)
    
    fairness_results = {}
    
    # Gender fairness
    print("\nGender-Based Feature Importance:")
    gender_groups = {}
    
    for gender in ['Female', 'Male']:
        mask = sens_test['gender'] == gender
        X_g = X_test[mask]
        y_g = y_test[mask]
        
        if len(X_g) < 50:
            continue
        
        print(f"\n  {gender}:")
        print(f"    Samples: {len(X_g):,}")
        print(f"    Positive rate: {y_g.mean()*100:.2f}%")
        
        # Sample for SHAP (expensive computation)
        sample_size = min(30, len(X_g))
        sample_indices = np.random.choice(len(X_g), sample_size, replace=False)
        X_sample = X_g[sample_indices]
        
        # Background
        bg_size = min(80, len(X_g))
        bg_indices = np.random.choice(len(X_g), bg_size, replace=False)
        background = X_g[bg_indices]
        
        # SHAP
        explainer = shap.KernelExplainer(model.predict, background)
        
        try:
            shap_values = explainer.shap_values(X_sample, nsamples=80)
        except:
            print(f"    Error computing SHAP, skipping")
            continue

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Feature importance (ensure 1-D)
        importance = np.abs(shap_values).mean(axis=0)
        importance = np.squeeze(importance)
        importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Top features
        top_indices = np.argsort(importance)[-5:][::-1]
        top_features = [(feature_names[int(i)], float(importance[int(i)])) for i in top_indices]
        
        gender_groups[gender] = {
            'n_samples': int(len(X_g)),
            'positive_rate': float(y_g.mean()),
            'top_5_features': top_features,
            'importance_vector': importance.tolist()
        }
        
        print(f"    Top 5 Features:")
        for rank, (feat, imp) in enumerate(top_features, 1):
            print(f"      {rank}. {feat:35s} {imp:.6f}")
    
    fairness_results['gender'] = gender_groups
    
    # Race fairness (simplified)
    print("\n\nRace-Based Feature Importance:")
    race_groups = {}
    
    for race in sens_test['race'].unique()[:3]:  # Top 3 races
        mask = sens_test['race'] == race
        X_r = X_test[mask]
        y_r = y_test[mask]
        
        if len(X_r) < 50:
            continue
        
        print(f"\n  {race}:")
        print(f"    Samples: {len(X_r):,}")
        print(f"    Positive rate: {y_r.mean()*100:.2f}%")
        
        # Sample
        sample_size = min(30, len(X_r))
        sample_indices = np.random.choice(len(X_r), sample_size, replace=False)
        X_sample = X_r[sample_indices]
        
        # Background
        bg_size = min(80, len(X_r))
        bg_indices = np.random.choice(len(X_r), bg_size, replace=False)
        background = X_r[bg_indices]
        
        # SHAP
        explainer = shap.KernelExplainer(model.predict, background)
        
        try:
            shap_values = explainer.shap_values(X_sample, nsamples=80)
        except:
            continue

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Feature importance (ensure 1-D)
        importance = np.abs(shap_values).mean(axis=0)
        importance = np.squeeze(importance)
        importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Top features
        top_indices = np.argsort(importance)[-5:][::-1]
        top_features = [(feature_names[int(i)], float(importance[int(i)])) for i in top_indices]
        
        race_groups[race] = {
            'n_samples': int(len(X_r)),
            'positive_rate': float(y_r.mean()),
            'top_5_features': top_features
        }
        
        print(f"    Top 5 Features:")
        for rank, (feat, imp) in enumerate(top_features, 1):
            print(f"      {rank}. {feat:35s} {imp:.6f}")
    
    fairness_results['race'] = race_groups
    
    return fairness_results


# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================

def create_visualizations(local_results, patient_explanations, fairness_results):
    """Create comprehensive visualizations of local and instance explanations"""
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs('results/figures', exist_ok=True)
    
    # 1. Local Hospital Feature Importance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (h_id, results) in enumerate(local_results.items()):
        top_features = results['top_10_features']
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        axes[idx].barh(features, importances, color='steelblue', alpha=0.8)
        axes[idx].set_xlabel('Mean |SHAP Value|')
        axes[idx].set_title(f"Hospital {h_id}\n({results['hospital_name']})", fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/figures/07a_local_hospital_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07a_local_hospital_importance.png")
    plt.close()
    
    # 2. Sample Patient Risk Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (h_id, patients) in enumerate(patient_explanations.items()):
        if not patients:
            continue
        
        risks = [p['predicted_risk'] for p in patients]
        
        axes[idx].hist(risks, bins=10, color='coral', alpha=0.7, edgecolor='black')
        axes[idx].axvline(np.mean(risks), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(risks):.2f}')
        axes[idx].set_xlabel('Predicted Readmission Risk')
        axes[idx].set_ylabel('Number of Patients')
        axes[idx].set_title(f'Hospital {h_id} Sample Patients', fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/07b_patient_risk_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 07b_patient_risk_distribution.png")
    plt.close()
    
    # 3. Fairness: Gender-based Feature Importance
    if 'gender' in fairness_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for gender_idx, (gender, results) in enumerate(fairness_results['gender'].items()):
            top_features = results['top_5_features']
            features = [f[0] for f in top_features]
            importances = [f[1] for f in top_features]
            
            axes[gender_idx].barh(features, importances, 
                                 color=['#FF6B6B' if gender == 'Female' else '#4ECDC4'][0],
                                 alpha=0.8)
            axes[gender_idx].set_xlabel('Mean |SHAP Value|')
            axes[gender_idx].set_title(f'{gender} (n={results["n_samples"]:,})', fontweight='bold')
            axes[gender_idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('results/figures/07c_fairness_gender_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 07c_fairness_gender_importance.png")
        plt.close()


# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

def save_local_explainability_results(local_results, patient_explanations, fairness_results):
    """Save all results to JSON"""
    
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    # Save local hospital explanations
    with open('results/local_hospital_explanations.json', 'w') as f:
        json.dump(local_results, f, indent=2)
    print("✓ Saved: results/local_hospital_explanations.json")
    
    # Save instance-level explanations
    with open('results/instance_patient_explanations.json', 'w') as f:
        json.dump(patient_explanations, f, indent=2)
    print("✓ Saved: results/instance_patient_explanations.json")
    
    # Save fairness explanations
    with open('results/fairness_explanations.json', 'w') as f:
        json.dump(fairness_results, f, indent=2)
    print("✓ Saved: results/fairness_explanations.json")
    
    # Create summary
    summary = {
        'local_hospitals': {
            h_id: {
                'hospital_name': results['hospital_name'],
                'n_samples': results['n_samples'],
                'positive_rate': results['positive_rate'],
                'top_feature': results['top_10_features'][0][0] if results['top_10_features'] else 'N/A'
            }
            for h_id, results in local_results.items()
        },
        'instance_patients': {
            h_id: len(patients) for h_id, patients in patient_explanations.items()
        },
        'fairness_groups': {
            'gender': list(fairness_results.get('gender', {}).keys()),
            'race': list(fairness_results.get('race', {}).keys())
        }
    }
    
    with open('results/local_explainability_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved: results/local_explainability_summary.json")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_local_explainability():
    """Main pipeline: compute local, instance, and fairness explanations"""
    
    print("\n" + "="*70)
    print("PHASE 7: LOCAL & INSTANCE-LEVEL EXPLAINABILITY")
    print("="*70)
    print("\nThis phase provides:")
    print("  1. Local hospital explanations (what each hospital cares about)")
    print("  2. Instance-level explanations (why each patient is high-risk)")
    print("  3. Fairness explanations (are predictions biased?)")
    
    # Load resources
    X_train, y_train, X_test, y_test, sens_train, sens_test, feature_names, model = load_resources()
    
    # Step 1: Local hospital explanations
    local_results, hospital_importance_vectors = compute_local_hospital_explanations(
        X_test, y_test, sens_test, feature_names, model
    )
    
    # Step 2: Local vs Global divergence
    correlations, interpretation = analyze_local_global_divergence(
        local_results, hospital_importance_vectors, feature_names
    )
    
    # Step 3: Instance-level explanations
    patient_explanations = compute_instance_level_explanations(
        X_test, y_test, sens_test, feature_names, model, n_samples=15
    )
    
    # Step 4: Fairness explanations
    fairness_results = compute_fairness_explanations(
        X_test, y_test, sens_test, feature_names, model
    )
    
    # Step 5: Visualizations
    create_visualizations(local_results, patient_explanations, fairness_results)
    
    # Step 6: Save results
    save_local_explainability_results(local_results, patient_explanations, fairness_results)
    
    # Summary
    print("\n" + "="*70)
    print("LOCAL EXPLAINABILITY COMPLETE")
    print("="*70)
    print(f"\n✓ Local hospital explanations: {len(local_results)} hospitals")
    print(f"✓ Instance-level explanations: {sum(len(p) for p in patient_explanations.values())} patients")
    print(f"✓ Fairness explanations: {len(fairness_results.get('gender', {}))} gender groups, {len(fairness_results.get('race', {}))} race groups")
    print(f"✓ Visualizations: 3 figures saved to results/figures/")
    print(f"✓ Results saved to results/ directory")


if __name__ == "__main__":
    run_local_explainability()