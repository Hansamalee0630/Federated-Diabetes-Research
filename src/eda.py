import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_raw_data(path):
    """Load raw UCI diabetes dataset"""
    print("Loading raw data...")
    df = pd.read_csv(path, na_values='?')
    
    # CRITICAL FIX: Remove rows with Unknown/Invalid gender
    # These (~3 rows in UCI dataset) can crash binary fairness calculations
    n_before = len(df)
    df = df[df['gender'] != 'Unknown/Invalid']
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Removed {n_dropped} rows with Unknown/Invalid gender")
    
    return df

def analyze_missing_values(df):
    """Analyze missing value patterns"""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(missing_df[missing_df['Missing_Percentage'] > 0])
    return missing_df

def analyze_target_distribution(df):
    """Analyze readmission distribution"""
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS (RAW DATA)")
    print("="*60)
    
    print("\nReadmitted Value Counts:")
    print(df['readmitted'].value_counts(dropna=False))
    
    readmit_counts = df['readmitted'].value_counts()
    readmit_pct = df['readmitted'].value_counts(normalize=True) * 100
    
    print(f"\nTarget Variable Distribution:")
    print(f"  <30 (Readmitted): {readmit_counts.get('<30', 0)} ({readmit_pct.get('<30', 0):.2f}%)")
    print(f"  >30 (Readmitted later): {readmit_counts.get('>30', 0)} ({readmit_pct.get('>30', 0):.2f}%)")
    print(f"  No (Not readmitted): {readmit_counts.get('No', 0)} ({readmit_pct.get('No', 0):.2f}%)")
    
    print(f"\n Raw Class Balance: {readmit_pct.get('<30', 0):.2f}% positive")
    print(f"    → This is HIGHLY IMBALANCED")
    print(f"    → Solution: Balanced data processing (stratified undersampling)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    readmit_counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'coral', 'lightgreen'])
    axes[0].set_title('Readmission Count Distribution (3-class)')
    axes[0].set_ylabel('Number of Patients')
    axes[0].tick_params(axis='x', rotation=45)
    
    readmit_pct.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Readmission Percentage Distribution')
    
    plt.tight_layout()
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/01_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_demographics(df):
    """Analyze demographic distributions"""
    print("\n" + "="*60)
    print("DEMOGRAPHIC ANALYSIS (RAW DATA)")
    print("="*60)
    
    print("\nRace Distribution:")
    print(df['race'].value_counts())
    
    print("\nGender Distribution:")
    print(df['gender'].value_counts())
    
    print("\nAge Distribution:")
    print(df['age'].value_counts().sort_index())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    df['race'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Race Distribution')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    df['gender'].value_counts().plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Gender Distribution')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    df['age'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Age Distribution')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/02_demographics.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_readmission_by_demographics(df):
    """Analyze readmission rates by demographic groups (BASELINE)"""
    print("\n" + "="*60)
    print("FAIRNESS BASELINE (Before Balancing & Model)")
    print("="*60)
    
    df_temp = df.copy()
    df_temp['readmitted_binary'] = (df_temp['readmitted'] == '<30').astype(int)
    
    print("\nReadmission Rate by Race:")
    race_readmit = df_temp.groupby('race')['readmitted_binary'].agg(['sum', 'count', 'mean'])
    race_readmit.columns = ['Readmitted_Count', 'Total', 'Readmission_Rate']
    print(race_readmit)
    
    print("\nReadmission Rate by Gender:")
    gender_readmit = df_temp.groupby('gender')['readmitted_binary'].agg(['sum', 'count', 'mean'])
    gender_readmit.columns = ['Readmitted_Count', 'Total', 'Readmission_Rate']
    print(gender_readmit)
    
    print("\nReadmission Rate by Age Group:")
    age_readmit = df_temp.groupby('age')['readmitted_binary'].agg(['sum', 'count', 'mean'])
    age_readmit.columns = ['Readmitted_Count', 'Total', 'Readmission_Rate']
    print(age_readmit)
    
    # Calculate fairness gaps
    gender_rates = gender_readmit['Readmission_Rate'].values
    gender_gap = gender_rates.max() - gender_rates.min()
    
    race_rates = race_readmit['Readmission_Rate'].values
    race_gap = race_rates.max() - race_rates.min()
    
    print("\n" + "-"*60)
    print("FAIRNESS GAP BASELINE (Raw Data - No Balancing):")
    print(f"  Gender Fairness Gap: {gender_gap*100:.2f}%")
    print(f"  Race Fairness Gap: {race_gap*100:.2f}%")
    print(f"\n  → These gaps should NOT INCREASE after model training")
    print(f"  → Fairness audit will compare against these baselines")
    print("-"*60)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gender_readmit['Readmission_Rate'].plot(kind='bar', ax=axes[0], color='#FF6B6B', width=0.6)
    axes[0].set_title('Readmission Rate by Gender (BASELINE)', fontweight='bold')
    axes[0].set_ylabel('Readmission Rate')
    axes[0].axhline(y=df_temp['readmitted_binary'].mean(), color='blue', linestyle='--', label='Overall Mean')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(True, alpha=0.3)
    
    race_readmit['Readmission_Rate'].plot(kind='bar', ax=axes[1], color='#4ECDC4', width=0.6)
    axes[1].set_title('Readmission Rate by Race (BASELINE)', fontweight='bold')
    axes[1].set_ylabel('Readmission_Rate')
    axes[1].axhline(y=df_temp['readmitted_binary'].mean(), color='blue', linestyle='--', label='Overall Mean')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/03_fairness_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_eda():
    """Run complete EDA"""
    print("\n" + "="*70)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    data_path = Path(__file__).resolve().parent.parent / 'datasets' / 'diabetes_130' / 'raw' / 'diabetic_data.csv'
    if not data_path.exists():
        # Backwards-compatible fallback for other setups
        data_path = Path('data/diabetic_data.csv')

    df = load_raw_data(str(data_path))
    print(f"✓ Loaded data: {df.shape}")
    
    analyze_missing_values(df)
    analyze_target_distribution(df)
    analyze_demographics(df)
    analyze_readmission_by_demographics(df)
    
    print("\n" + "="*70)
    print("EDA COMPLETE!")
    print("="*70)
    print("Files saved to results/figures/")

if __name__ == "__main__":
    run_eda()