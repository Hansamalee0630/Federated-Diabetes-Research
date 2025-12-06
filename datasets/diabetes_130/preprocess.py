import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

# CONFIGURATION
RAW_DATA_PATH = "datasets/diabetes_130/diabetic_data.csv"
OUTPUT_DIR = "datasets/diabetes_130/processed"
NUM_CLIENTS = 3  # Simulating 3 hospitals

def load_and_clean_data():
    print("Loading Diabetes 130-US dataset...")
    # 1. Load Data (replace '?' with NaN)
    df = pd.read_csv(RAW_DATA_PATH, na_values='?')
    
    # 2. Basic Cleaning: Drop columns with too many missing values
    df = df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)
    df = df.dropna(subset=['race', 'diag_1', 'diag_2', 'diag_3'])

    # 3. COMPONENT 4 TARGETS: Generate Comorbidity Labels from ICD-9 Codes
    # Hypertension: 401.xx - 405.xx
    # Heart Failure: 428.xx
    print("Generating Comorbidity Labels (Hypertension & Heart Failure)...")
    
    def check_disease(row, prefixes):
        # Check all 3 diagnosis columns
        for col in ['diag_1', 'diag_2', 'diag_3']:
            val = str(row[col])
            if any(val.startswith(p) for p in prefixes):
                return 1
            # Handle diabetes-specific codes that imply complications if needed
        return 0

    # ICD-9 prefixes for Hypertension (401-405)
    htn_prefixes = ['401', '402', '403', '404', '405']
    # ICD-9 prefixes for Heart Failure (428)
    hf_prefixes = ['428']

    df['target_hypertension'] = df.apply(lambda x: check_disease(x, htn_prefixes), axis=1)
    df['target_heart_failure'] = df.apply(lambda x: check_disease(x, hf_prefixes), axis=1)

    # 4. COMPONENT 2 TARGET: Readmission (<30 days = 1, else 0)
    print("Generating Readmission Labels...")
    df['target_readmission'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # 5. Feature Engineering (Simplify for demo)
    # Keep essential features
    feature_cols = ['race', 'gender', 'age', 'time_in_hospital', 'num_lab_procedures', 
                    'num_procedures', 'num_medications', 'number_diagnoses', 
                    'A1Cresult', 'insulin', 'change', 'diabetesMed']
    
    X = df[feature_cols]
    # Gather all targets
    y = df[['target_readmission', 'target_hypertension', 'target_heart_failure']]

    return X, y

def split_for_clients(X, y):
    print(f"Splitting data for {NUM_CLIENTS} federated clients...")
    
    # Preprocessing: OneHotEncode Categoricals + Scale Numerics
    # (In real FL, scaling happens locally, but we do it here for simulation simplicity)
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Simple IID Split (Shuffle and chunk)
    # For Non-IID (Component 4), you would split by 'race' or 'gender' instead
    total_samples = len(X)
    chunk_size = total_samples // NUM_CLIENTS
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i in range(NUM_CLIENTS):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        
        X_client = X.iloc[start_idx:end_idx]
        y_client = y.iloc[start_idx:end_idx]
        
        # Save to CSV
        X_client.to_csv(f"{OUTPUT_DIR}/client_{i}_X.csv", index=False)
        y_client.to_csv(f"{OUTPUT_DIR}/client_{i}_y.csv", index=False)
        print(f"  -> Saved Client {i} data: {len(X_client)} samples")

if __name__ == "__main__":
    X, y = load_and_clean_data()
    split_for_clients(X, y)
    print("Preprocessing Complete. Data ready in /processed folder.")
