
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pickle
import json
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: LOAD RAW DATA AND CREATE BALANCED DATASET
# ============================================================================

def load_raw_data(path):
    """Load raw UCI diabetes dataset."""
    print("Loading raw data...")
    df = pd.read_csv(path, na_values='?', low_memory=False)
    return df


def create_balanced_dataset_stratified(df):
    """
    Create balanced dataset using STRATIFIED UNDERSAMPLING.
    
    Strategy:
    - Keep ALL <30 samples (11,357)
    - Undersample NO and >30 STRATIFIED by:
      * Gender (preserve 54% F, 46% M)
      * Hospital (preserve 29%, 13%, 55%)
      * Age (preserve age distributions)
    
    Result: ~22,714 samples (50% positive, 50% negative)
    """
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET (Stratified)")
    print("="*70)
    
    # Step 1: Create hospital IDs (needed for stratification)
    def group_diagnosis_codes(code):
        if pd.isna(code):
            return 'Other'
        try:
            code_str = str(code).strip()
            if code_str.startswith('V') or code_str.startswith('E'):
                return 'Other'
            code_num = float(code_str)
            if 390 <= code_num <= 459 or code_num == 785:
                return 'Circulatory'
            elif 460 <= code_num <= 519 or code_num == 786:
                return 'Respiratory'
            elif 520 <= code_num <= 579 or code_num == 787:
                return 'Digestive'
            elif 250 <= code_num < 251:
                return 'Diabetes'
            elif 800 <= code_num <= 999:
                return 'Injury'
            elif 710 <= code_num <= 739:
                return 'Musculoskeletal'
            elif 580 <= code_num <= 629 or code_num == 788:
                return 'Genitourinary'
            elif 140 <= code_num <= 239:
                return 'Neoplasms'
            else:
                return 'Other'
        except:
            return 'Other'
    
    # Assign hospital IDs
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(group_diagnosis_codes)
    
    hospital_id = np.zeros(len(df), dtype=int)
    circ_mask = df['diag_1'].isin(['Circulatory']).values
    hospital_id[circ_mask] = 1
    metabolic_mask = df['diag_1'].isin(['Diabetes', 'Genitourinary']).values
    hospital_id[metabolic_mask] = 2
    hospital_id[hospital_id == 0] = 3
    df['hospital_id'] = hospital_id
    
    # Step 2: Create binary target
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    
    # Step 3: Separate positive and negative samples
    positive = df[df['readmitted_binary'] == 1].copy()
    negative = df[df['readmitted_binary'] == 0].copy()
    
    print(f"\nOriginal distribution:")
    print(f"  Positive (<30): {len(positive):,} ({len(positive)/len(df)*100:.2f}%)")
    print(f"  Negative (>30 or NO): {len(negative):,} ({len(negative)/len(df)*100:.2f}%)")
    
    # Step 4: Stratified undersampling of negative class
    # Undersample to match positive class size
    target_size = len(positive)
    
    print(f"\nStratified Undersampling:")
    print(f"  Target size: {target_size:,} (per class)")
    
    # Stratify by [gender × hospital] to preserve distributions
    negative_sampled = negative.groupby(
        ['gender', 'hospital_id'],
        group_keys=False
    ).apply(
        lambda x: x.sample(
            n=int(len(x) * (target_size / len(negative))),
            random_state=42,
            replace=False
        )
    )
    
    # Step 5: Combine positive and negative for balanced dataset
    balanced_df = pd.concat([positive, negative_sampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced distribution:")
    print(f"  Positive (<30): {(balanced_df['readmitted_binary'] == 1).sum():,} " 
          f"({(balanced_df['readmitted_binary'] == 1).mean()*100:.2f}%)")
    print(f"  Negative (>30 or NO): {(balanced_df['readmitted_binary'] == 0).sum():,} "
          f"({(balanced_df['readmitted_binary'] == 0).mean()*100:.2f}%)")
    
    # Step 6: Verify distributions are preserved
    print(f"\nVerifying Distributions Preserved:")
    
    print(f"\n  Gender Distribution:")
    for gender in balanced_df['gender'].unique():
        if pd.notna(gender):
            pct = (balanced_df['gender'] == gender).mean() * 100
            print(f"    {gender}: {pct:.2f}%")
    
    print(f"\n  Hospital Distribution:")
    for h in sorted(balanced_df['hospital_id'].unique()):
        pct = (balanced_df['hospital_id'] == h).mean() * 100
        count = (balanced_df['hospital_id'] == h).sum()
        print(f"    Hospital {h}: {count:,} ({pct:.2f}%)")
    
    print(f"\n  Gender Fairness (Baseline):")
    for gender in ['Female', 'Male']:
        readmit_rate = (balanced_df[balanced_df['gender'] == gender]['readmitted_binary']).mean()
        print(f"    {gender} readmission rate: {readmit_rate*100:.2f}%")
    
    return balanced_df


# ============================================================================
# STEP 2: PREPROCESS BALANCED DATA
# ============================================================================

def preprocess_balanced_data(balanced_df):
    """
    Preprocess balanced dataset (same as original preprocessing).
    """
    print("\n" + "="*70)
    print("PREPROCESSING BALANCED DATA")
    print("="*70)
    
    # Clean data
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id']
    available_drop_cols = [col for col in drop_cols if col in balanced_df.columns]
    balanced_df = balanced_df.drop(columns=available_drop_cols)
    
    if 'discharge_disposition_id' in balanced_df.columns:
        balanced_df = balanced_df[
            ~balanced_df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])
        ]
    
    # Handle missing demographics
    if 'race' in balanced_df.columns:
        balanced_df['race'] = balanced_df['race'].fillna('Unknown')
    if 'gender' in balanced_df.columns:
        # CRITICAL FIX: Drop Unknown/Invalid gender rows entirely
        # These (~3 rows) can crash binary fairness calculations
        n_before = len(balanced_df)
        balanced_df = balanced_df[balanced_df['gender'] != 'Unknown/Invalid']
        n_dropped = n_before - len(balanced_df)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} rows with Unknown/Invalid gender")
    
    # Preserve sensitive attributes
    sensitive_attrs = balanced_df[['hospital_id', 'race', 'gender', 'age']].copy()
    
    # Define features
    medication_features = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citogliptin', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change'
    ]
    
    categorical_features = [
        'race', 'gender', 'age', 'admission_type_id',
        'discharge_disposition_id', 'admission_source_id',
        'diag_1', 'diag_2', 'diag_3',
        'max_glu_serum', 'A1Cresult', 'diabetesMed'
    ] + medication_features
    
    numerical_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]
    
    categorical_features = [col for col in categorical_features if col in balanced_df.columns]
    numerical_features = [col for col in numerical_features if col in balanced_df.columns]
    
    X = balanced_df.drop(['readmitted_binary', 'patient_nbr', 'hospital_id', 'readmitted'],
                          axis=1, errors='ignore')
    y = balanced_df['readmitted_binary']
    
    # CRITICAL FIX: Hospital-wise NUMERIC scaling (LOCAL, NOT GLOBAL)
    # ===========================================================
    # In true Federated Learning, the server CANNOT see all data.
    # 
    # KEY INSIGHT: We can't fit OneHotEncoder per hospital because:
    # - Hospital A might have categories that Hospital B doesn't see
    # - This creates mismatched feature dimensions (ML incompatible)
    # 
    # SOLUTION: Two-tier preprocessing:
    # 1. OneHotEncoder: Global (fit once on all data for consistency)
    # 2. StandardScaler: Local (fit per hospital for privacy)
    # 
    # This balances: (1) FL correctness with (2) ML compatibility
    # Result: More realistic FL + locally scaled numerical features
    
    print("Building global categorical preprocessor (feature dimensions)...")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
    ])
    
    # Create feature names from all data to ensure consistent dimensions
    X_cat_transformed = categorical_transformer.fit_transform(X[categorical_features])
    onehot_encoder = categorical_transformer.named_steps['onehot']
    cat_feature_names = list(onehot_encoder.get_feature_names_out(categorical_features))
    feature_names = numerical_features + cat_feature_names
    
    # Hospital-wise train-test split with LOCAL NUMERIC scaling
    print("\nPerforming HOSPITAL-WISE train-test split with LOCAL numeric scaling...")
    print("(Categorical: global for consistency | Numeric: local for privacy)\n")
    
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    sens_train_list, sens_test_list = [], []
    hospital_stats = {}
    hospital_scalers = {}  # Store each hospital's NUMERIC scaler
    
    for h in [1, 2, 3]:
        mask = (balanced_df['hospital_id'] == h).values
        
        # Extract hospital data (BEFORE any scaling)
        X_h_raw = X[mask]
        y_h = y[mask].values
        sens_h = sensitive_attrs[mask]
        
        # Split first, then scale (train scaler on training data only)
        if len(y_h) > 10:
            X_tr_raw, X_te_raw, y_tr, y_te, sens_tr, sens_te = train_test_split(
                X_h_raw, y_h, sens_h,
                test_size=0.2, random_state=42, stratify=y_h
            )
        else:
            X_tr_raw, X_te_raw, y_tr, y_te, sens_tr, sens_te = X_h_raw, X_h_raw, y_h, y_h, sens_h, sens_h
        
        # Step 1: Apply categorical transformation (global, for consistency)
        X_tr_cat = categorical_transformer.transform(X_tr_raw[categorical_features])
        X_te_cat = categorical_transformer.transform(X_te_raw[categorical_features])
        
        # Step 2: Apply hospital-specific NUMERIC scaling
        # Fit scaler on hospital's training data ONLY
        numeric_scaler = StandardScaler()
        
        # Prepare numeric features
        X_tr_num = X_tr_raw[numerical_features].values
        X_te_num = X_te_raw[numerical_features].values
        
        # Impute missing values per hospital
        imputer = SimpleImputer(strategy='median')
        X_tr_num = imputer.fit_transform(X_tr_num)
        X_te_num = imputer.transform(X_te_num)
        
        # Scale numeric features on hospital's training data (LOCAL scaling)
        X_tr_num = numeric_scaler.fit_transform(X_tr_num)
        X_te_num = numeric_scaler.transform(X_te_num)
        
        # Combine categorical and numeric
        X_tr = np.hstack([X_tr_num, X_tr_cat])
        X_te = np.hstack([X_te_num, X_te_cat])
        
        # Store hospital's numeric scaler for reference
        hospital_scalers[h] = numeric_scaler
        
        # Add to aggregate lists
        X_train_list.append(X_tr)
        X_test_list.append(X_te)
        y_train_list.append(y_tr)
        y_test_list.append(y_te)
        sens_train_list.append(sens_tr)
        sens_test_list.append(sens_te)
        
        print(f"  Hospital {h}: Train={len(y_tr):,}, Test={len(y_te):,} | "
              f"Pos Rate: {y_tr.mean():.4f} | LOCAL numeric scaling applied")
        
        hospital_stats[h] = {
            'n_train': int(len(y_tr)),
            'pos_rate_train': float(y_tr.mean()),
            'n_test': int(len(y_te)),
            'pos_rate_test': float(y_te.mean())
        }
    
    # Combine all hospitals
    X_train_final = np.vstack(X_train_list)
    X_test_final = np.vstack(X_test_list)
    y_train_final = np.concatenate(y_train_list)
    y_test_final = np.concatenate(y_test_list)
    sens_train_final = pd.concat(sens_train_list).reset_index(drop=True)
    sens_test_final = pd.concat(sens_test_list).reset_index(drop=True)
    
    return (X_train_final, y_train_final, X_test_final, y_test_final,
            sens_train_final, sens_test_final, feature_names, hospital_scalers, hospital_stats)


# ============================================================================
# STEP 3: SAVE BALANCED DATA
# ============================================================================

def save_balanced_data(X_train, y_train, X_test, y_test,
                       sens_train, sens_test, feature_names,
                       hospital_scalers, hospital_stats):
    """Save balanced dataset with hospital-specific numeric scalers."""
    print("\n" + "="*70)
    print("SAVING BALANCED DATA (with Hospital-Specific Numeric Scaling)")
    print("="*70)
    
    output_dir = Path('data/processed_data_balanced')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'X_train_balanced.npy', X_train.astype(np.float32))
    np.save(output_dir / 'y_train_balanced.npy', y_train.astype(np.float32))
    np.save(output_dir / 'X_test_balanced.npy', X_test.astype(np.float32))
    np.save(output_dir / 'y_test_balanced.npy', y_test.astype(np.float32))
    
    sens_train.to_csv(output_dir / 'sensitive_attrs_train_balanced.csv', index=False)
    sens_test.to_csv(output_dir / 'sensitive_attrs_test_balanced.csv', index=False)
    
    with open(output_dir / 'feature_names_balanced.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    with open(output_dir / 'hospital_stats_balanced.json', 'w') as f:
        json.dump(hospital_stats, f, indent=2)
    
    # Save hospital-specific numeric scalers for reference
    # NOTE: In true FL, each hospital keeps its own scaler locally
    with open(output_dir / 'hospital_scalers.pkl', 'wb') as f:
        pickle.dump(hospital_scalers, f)
    
    print(f"\n✓ All balanced data saved to: {output_dir}")
    print(f"✓ Hospital-specific numeric scalers saved (for reference/debugging)")
    print(f"\nFinal Balanced Dataset (with LOCAL numeric scaling):")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Train positive rate: {y_train.mean()*100:.2f}%")
    print(f"  Test positive rate: {y_test.mean()*100:.2f}%")
    print(f"\nFederated Learning Note:")
    print(f"  Categorical features: global (for consistent feature dimensions)")
    print(f"  Numeric features: local per hospital (preserves privacy)")
    print(f"  Each hospital scales its own numerical features independently")
    print(f"  Server never sees hospital-level numeric statistics")
    
    return output_dir


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("INTELLIGENT DATA BALANCING FOR FEDERATED LEARNING")
    print("="*70)
    
    # Step 1: Load raw data
    data_path = Path(__file__).resolve().parent.parent / 'datasets' / 'diabetes_130' / 'raw' / 'diabetic_data.csv'
    if not data_path.exists():
        # Backwards-compatible fallback for other setups
        data_path = Path('data/diabetic_data.csv')

    df = load_raw_data(str(data_path))
    print(f"Loaded raw data: {df.shape}")
    
    # Step 2: Create balanced dataset
    balanced_df = create_balanced_dataset_stratified(df)
    
    # Step 3: Preprocess balanced data (with LOCAL numeric scaling)
    X_train, y_train, X_test, y_test, sens_train, sens_test, \
    feature_names, hospital_scalers, hospital_stats = preprocess_balanced_data(balanced_df)
    
    # Step 4: Save balanced data
    output_dir = save_balanced_data(
        X_train, y_train, X_test, y_test,
        sens_train, sens_test, feature_names,
        hospital_scalers, hospital_stats
    )
    
    print("\n" + "="*70)
    print("BALANCED DATASET CREATION COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
 


if __name__ == "__main__":
    main()