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

def load_raw_data(path):
    print("Loading raw data...")
    df = pd.read_csv(path, na_values='?', low_memory=False)
    
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
    except: return 'Other'

def preprocess_pipeline():
    print("\n" + "="*70)
    print("DATA PREPROCESSING (PERFECT BALANCING & NO LEAKAGE)")
    print("="*70)
    
    # 1. Load Data
    data_path = Path('data/diabetic_data.csv')
    if not data_path.exists(): 
        data_path = Path('diabetic_data.csv')
    
    df = load_raw_data(str(data_path))
    
    # 2. Fix Leakage (Remove dead/hospice patients)
    if 'discharge_disposition_id' in df.columns:
        df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

    # 3. Create Target
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)

    # 4. Assign Hospital IDs
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(group_diagnosis_codes)
            
    hospital_id = np.zeros(len(df), dtype=int)
    hospital_id[df['diag_1'].isin(['Circulatory']).values] = 1
    hospital_id[df['diag_1'].isin(['Diabetes', 'Genitourinary']).values] = 2
    hospital_id[hospital_id == 0] = 3
    df['hospital_id'] = hospital_id
    
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')

    # 5. EXACT STRATIFIED BALANCING (The Fix)
    print("\nExecuting precise 50/50 Balancing...")
    pos_df = df[df['readmitted_binary'] == 1].copy()
    neg_df = df[df['readmitted_binary'] == 0].copy()
    
    # We want exactly as many negatives as positives
    n_pos = len(pos_df)
    
    # Stratify by gender and hospital
    neg_df['stratum'] = neg_df['hospital_id'].astype(str) + "_" + neg_df['gender'].astype(str)
    
    # Sample negatives exactly to match positives
    neg_sampled, _ = train_test_split(neg_df, train_size=n_pos, stratify=neg_df['stratum'], random_state=42)
    neg_sampled = neg_sampled.drop(columns=['stratum'])
    
    balanced_df = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Dataset Size: {len(balanced_df)}")
    print(f"Positives (1): {sum(balanced_df['readmitted_binary'] == 1)}")
    print(f"Negatives (0): {sum(balanced_df['readmitted_binary'] == 0)}")
    assert sum(balanced_df['readmitted_binary'] == 1) == sum(balanced_df['readmitted_binary'] == 0), "Balancing Failed!"

    # 6. Feature Extraction (AGGRESSIVE CLEANING)
    print("\nDropping irrelevant/leaky ID columns...")
    
    # Base columns to drop
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'readmitted']
    
    # Aggressively find any 'discharge' or 'admission' related columns
    # This guarantees admission_type_id, admission_source_id, and discharge_disposition_id are caught
    id_cols = [col for col in balanced_df.columns if 'discharge' in col.lower() or 'admission' in col.lower()]
    drop_cols.extend(id_cols)
    
    # Drop them all safely
    cols_to_drop = list(set([c for c in drop_cols if c in balanced_df.columns]))
    print(f"  Dropping columns: {cols_to_drop}")
    balanced_df = balanced_df.drop(columns=cols_to_drop)
    
    sensitive_attrs = balanced_df[['hospital_id', 'race', 'gender', 'age']].copy()
    y = balanced_df['readmitted_binary'].values
    
    # Create X (features), making sure target and identifiers are removed
    X = balanced_df.drop(columns=['readmitted_binary', 'patient_nbr', 'hospital_id'], errors='ignore')

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # 7. Transformations
    print("Building global categorical preprocessor...")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    X_cat_transformed = categorical_transformer.fit_transform(X[categorical_features])
    cat_feature_names = list(categorical_transformer.named_steps['onehot'].get_feature_names_out(categorical_features))
    feature_names = numerical_features + cat_feature_names

    print("\nPerforming HOSPITAL-WISE train-test split with LOCAL scaling...")
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    sens_train_list, sens_test_list = [], []
    hospital_stats, hospital_scalers = {}, {}
    
    for h in [1, 2, 3]:
        mask = (balanced_df['hospital_id'] == h).values
        X_h_raw = X[mask]
        y_h = y[mask]
        sens_h = sensitive_attrs[mask]
        
        X_tr_raw, X_te_raw, y_tr, y_te, sens_tr, sens_te = train_test_split(
            X_h_raw, y_h, sens_h, test_size=0.2, random_state=42, stratify=y_h
        )
        
        X_tr_cat = categorical_transformer.transform(X_tr_raw[categorical_features])
        X_te_cat = categorical_transformer.transform(X_te_raw[categorical_features])
        
        numeric_scaler = StandardScaler()
        imputer = SimpleImputer(strategy='median')
        
        X_tr_num = numeric_scaler.fit_transform(imputer.fit_transform(X_tr_raw[numerical_features].values))
        X_te_num = numeric_scaler.transform(imputer.transform(X_te_raw[numerical_features].values))
        
        hospital_scalers[h] = numeric_scaler
        
        X_train_list.append(np.hstack([X_tr_num, X_tr_cat]))
        X_test_list.append(np.hstack([X_te_num, X_te_cat]))
        y_train_list.append(y_tr)
        y_test_list.append(y_te)
        sens_train_list.append(sens_tr)
        sens_test_list.append(sens_te)
        
        print(f" Hospital {h}: Train={len(y_tr)}, Test={len(y_te)} | Pos Rate: {y_tr.mean():.4f}")
        hospital_stats[h] = {'n_train': int(len(y_tr)), 'pos_rate_train': float(y_tr.mean())}

    X_train_final = np.vstack(X_train_list)
    X_test_final = np.vstack(X_test_list)
    y_train_final = np.concatenate(y_train_list)
    y_test_final = np.concatenate(y_test_list)
    sens_train_final = pd.concat(sens_train_list).reset_index(drop=True)
    sens_test_final = pd.concat(sens_test_list).reset_index(drop=True)

    print("\n" + "="*70)
    print("SAVING PROCESSED DATA")
    print("="*70)

    for folder in ['data/processed_data_balanced', 'data/processed_data']:
        os.makedirs(folder, exist_ok=True)
        suffix = '_balanced' if 'balanced' in folder else ''
        
        np.save(f'{folder}/X_train{suffix}.npy', X_train_final.astype(np.float32))
        np.save(f'{folder}/y_train{suffix}.npy', y_train_final.astype(np.float32))
        np.save(f'{folder}/X_test{suffix}.npy', X_test_final.astype(np.float32))
        np.save(f'{folder}/y_test{suffix}.npy', y_test_final.astype(np.float32))
        
        sens_train_final.to_csv(f'{folder}/sensitive_attrs_train{suffix}.csv', index=False)
        sens_test_final.to_csv(f'{folder}/sensitive_attrs_test{suffix}.csv', index=False)
        
        with open(f'{folder}/feature_names{suffix}.txt', 'w') as f:
            f.write('\n'.join(feature_names))

    with open('data/processed_data/hospital_stats.json', 'w') as f: json.dump(hospital_stats, f)
    with open('data/processed_data/hospital_scalers.pkl', 'wb') as f: pickle.dump(hospital_scalers, f)

    print(f"\n✓ Data fully processed. Global Positive Rate: {y_train_final.mean()*100:.2f}%")

if __name__ == "__main__":
    preprocess_pipeline()
