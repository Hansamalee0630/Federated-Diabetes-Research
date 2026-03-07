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
# STEP 1: LOAD RAW DATA AND BALANCE
# ============================================================================

def load_raw_data(path):
    """Load raw UCI diabetes dataset."""
    print("Loading raw data...")
    df = pd.read_csv(path, na_values='?', low_memory=False)
    
    # CRITICAL FIX: Drop Unknown/Invalid gender to prevent fairness metric crashes
    n_before = len(df)
    df = df[df['gender'] != 'Unknown/Invalid']
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with Unknown/Invalid gender")
    
    return df

def group_diagnosis_codes(code):
    if pd.isna(code): return 'Other'
    try:
        code_str = str(code).strip()
        if code_str.startswith('V') or code_str.startswith('E'): return 'Other'
        code_num = float(code_str)
        if 390 <= code_num <= 459 or code_num == 785: return 'Circulatory'
        elif 460 <= code_num <= 519 or code_num == 786: return 'Respiratory'
        elif 520 <= code_num <= 579 or code_num == 787: return 'Digestive'
        elif 250 <= code_num < 251: return 'Diabetes'
        elif 800 <= code_num <= 999: return 'Injury'
        elif 710 <= code_num <= 739: return 'Musculoskeletal'
        elif 580 <= code_num <= 629 or code_num == 788: return 'Genitourinary'
        elif 140 <= code_num <= 239: return 'Neoplasms'
        else: return 'Other'
    except: 
        return 'Other'

def create_balanced_dataset_stratified(df):
    """
    Create balanced dataset using STRATIFIED UNDERSAMPLING.
    Keeps all <30 samples. Undersamples negative class to match.
    """
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET (Stratified)")
    print("="*70)
    
    # 1. FIX LEAKAGE: Drop terminal patients (Hospice/Deceased cannot be readmitted)
    if 'discharge_disposition_id' in df.columns:
        n_pre = len(df)
        df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
        print(f"  Removed {n_pre - len(df)} terminal patients (prevents data leakage)")

    # 2. Assign hospital IDs based on diagnosis heterogeneity
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
    
    # 3. Create binary target
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    
    # 4. Separate positive and negative samples
    positive = df[df['readmitted_binary'] == 1].copy()
    negative = df[df['readmitted_binary'] == 0].copy()
    
    print(f"\nOriginal valid distribution:")
    print(f"  Positive (<30): {len(positive):,}")
    print(f"  Negative (>30 or NO): {len(negative):,}")
    
    # 5. Stratified undersampling
    target_size = len(positive)
    print(f"  Target size per class: {target_size:,}")
    
    # Stratify strictly by [gender AND hospital_id] to maintain fairness distributions
    negative_sampled = negative.groupby(
        ['gender', 'hospital_id'], group_keys=False
    ).apply(
        lambda x: x.sample(
            n=int(np.ceil(len(x) * (target_size / len(negative)))),
            random_state=42, replace=False
        )
    )
    
    # Trim excess if rounding overshoots
    if len(negative_sampled) > target_size:
        negative_sampled = negative_sampled.sample(n=target_size, random_state=42)
    
    balanced_df = pd.concat([positive, negative_sampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced distribution:")
    print(f"  Positive (<30): {(balanced_df['readmitted_binary'] == 1).sum():,} ({(balanced_df['readmitted_binary'] == 1).mean()*100:.1f}%)")
    print(f"  Negative (>30/NO): {(balanced_df['readmitted_binary'] == 0).sum():,} ({(balanced_df['readmitted_binary'] == 0).mean()*100:.1f}%)")
    
    return balanced_df

# ============================================================================
# STEP 2: PREPROCESS DATA
# ============================================================================

def preprocess_balanced_data(balanced_df):
    print("\n" + "="*70)
    print("PREPROCESSING BALANCED DATA")
    print("="*70)
    
    # Drop leaky and irrelevant columns completely
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'discharge_disposition_id']
    available_drop_cols = [col for col in drop_cols if col in balanced_df.columns]
    balanced_df = balanced_df.drop(columns=available_drop_cols)
    
    if 'race' in balanced_df.columns:
        balanced_df['race'] = balanced_df['race'].fillna('Unknown')
        
    sensitive_attrs = balanced_df[['hospital_id', 'race', 'gender', 'age']].copy()
    
    medication_features = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citogliptin', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change'
    ]
    
    categorical_features = [
        'race', 'gender', 'age', 'admission_type_id', 
        'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 
        'max_glu_serum', 'A1Cresult', 'diabetesMed'
    ] + medication_features
    
    numerical_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]
    
    categorical_features = [col for col in categorical_features if col in balanced_df.columns]
    numerical_features = [col for col in numerical_features if col in balanced_df.columns]
    
    X = balanced_df.drop(['readmitted_binary', 'patient_nbr', 'hospital_id', 'readmitted'], axis=1, errors='ignore')
    y = balanced_df['readmitted_binary']
    
    print("Building global categorical preprocessor (ensures equal feature dimensions)...")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
    ])
    
    # Fit globally to get uniform one-hot vector dimensions
    X_cat_transformed = categorical_transformer.fit_transform(X[categorical_features])
    onehot_encoder = categorical_transformer.named_steps['onehot']
    cat_feature_names = list(onehot_encoder.get_feature_names_out(categorical_features))
    feature_names = numerical_features + cat_feature_names
    
    print("\nPerforming HOSPITAL-WISE train-test split with LOCAL numeric scaling...")
    print("(Categorical: global for consistency | Numeric: local for privacy)\n")
    
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    sens_train_list, sens_test_list = [], []
    hospital_stats = {}
    hospital_scalers = {}
    
    for h in [1, 2, 3]:
        mask = (balanced_df['hospital_id'] == h).values
        X_h_raw = X[mask]
        y_h = y[mask].values
        sens_h = sensitive_attrs[mask]
        
        # Split first
        if len(y_h) > 10:
            X_tr_raw, X_te_raw, y_tr, y_te, sens_tr, sens_te = train_test_split(
                X_h_raw, y_h, sens_h, test_size=0.2, random_state=42, stratify=y_h
            )
        else:
            X_tr_raw, X_te_raw, y_tr, y_te, sens_tr, sens_te = X_h_raw, X_h_raw, y_h, y_h, sens_h, sens_h
            
        # Global categorical transform
        X_tr_cat = categorical_transformer.transform(X_tr_raw[categorical_features])
        X_te_cat = categorical_transformer.transform(X_te_raw[categorical_features])
        
        # LOCAL Numeric Scaling (Federated Learning Standard)
        numeric_scaler = StandardScaler()
        imputer = SimpleImputer(strategy='median')
        
        X_tr_num = imputer.fit_transform(X_tr_raw[numerical_features].values)
        X_te_num = imputer.transform(X_te_raw[numerical_features].values)
        
        X_tr_num = numeric_scaler.fit_transform(X_tr_num)
        X_te_num = numeric_scaler.transform(X_te_num)
        
        hospital_scalers[h] = numeric_scaler
        
        # Combine numerical and categorical features
        X_tr = np.hstack([X_tr_num, X_tr_cat])
        X_te = np.hstack([X_te_num, X_te_cat])
        
        X_train_list.append(X_tr)
        X_test_list.append(X_te)
        y_train_list.append(y_tr)
        y_test_list.append(y_te)
        sens_train_list.append(sens_tr)
        sens_test_list.append(sens_te)
        
        print(f"  Hospital {h}: Train={len(y_tr):,}, Test={len(y_te):,} | Pos Rate: {y_tr.mean():.4f}")
        hospital_stats[h] = {'n_train': int(len(y_tr)), 'pos_rate_train': float(y_tr.mean()), 'n_test': int(len(y_te)), 'pos_rate_test': float(y_te.mean())}
        
    X_train_final = np.vstack(X_train_list)
    X_test_final = np.vstack(X_test_list)
    y_train_final = np.concatenate(y_train_list)
    y_test_final = np.concatenate(y_test_list)
    sens_train_final = pd.concat(sens_train_list).reset_index(drop=True)
    sens_test_final = pd.concat(sens_test_list).reset_index(drop=True)
    
    return (X_train_final, y_train_final, X_test_final, y_test_final, 
            sens_train_final, sens_test_final, feature_names, hospital_scalers, hospital_stats)

def main():
    print("\n" + "="*70)
    print("INTELLIGENT DATA PREPROCESSING FOR FEDERATED LEARNING")
    print("="*70)

    # Locate Data
    data_path = Path('data/diabetic_data.csv')
    if not data_path.exists(): 
        data_path = Path('diabetic_data.csv')

    # Execute Pipeline
    df = load_raw_data(str(data_path))
    balanced_df = create_balanced_dataset_stratified(df)
    
    (X_train, y_train, X_test, y_test, sens_train, sens_test, 
     feature_names, hospital_scalers, hospital_stats) = preprocess_balanced_data(balanced_df)
    
    print("\n" + "="*70)
    print("SAVING PROCESSED DATA (FULLY BALANCED & NO LEAKAGE)")
    print("="*70)

    # Save to standard directories
    output_dir_bal = Path('data/processed_data_balanced')
    output_dir_std = Path('data/processed_data') 
    
    for folder in [output_dir_bal, output_dir_std]:
        folder.mkdir(parents=True, exist_ok=True)
        # Naming variables dynamically based on folder
        suffix = '_balanced' if 'balanced' in str(folder) else ''
        
        np.save(folder / f'X_train{suffix}.npy', X_train.astype(np.float32))
        np.save(folder / f'y_train{suffix}.npy', y_train.astype(np.float32))
        np.save(folder / f'X_test{suffix}.npy', X_test.astype(np.float32))
        np.save(folder / f'y_test{suffix}.npy', y_test.astype(np.float32))
        
        sens_train.to_csv(folder / f'sensitive_attrs_train{suffix}.csv', index=False)
        sens_test.to_csv(folder / f'sensitive_attrs_test{suffix}.csv', index=False)
        
        with open(folder / f'feature_names{suffix}.txt', 'w') as f:
            f.write('\n'.join(feature_names))

    # Save metadata
    with open('data/processed_data/hospital_stats.json', 'w') as f: 
        json.dump(hospital_stats, f, indent=2)

    with open('data/processed_data/hospital_scalers.pkl', 'wb') as f:
        pickle.dump(hospital_scalers, f)

    print("\n✓ Processing Complete. Data is perfectly balanced and leakage-free.")

if __name__ == "__main__":
    main()