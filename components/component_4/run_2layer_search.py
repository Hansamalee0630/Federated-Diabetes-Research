"""
Two-Layer Architecture Search
Finds the optimal 2-layer shared configuration
"""

import subprocess
import json
import pandas as pd
import sys
from datetime import datetime

# ============================================
# 2-LAYER SEARCH SPACE
# ============================================
CONFIGS = [
    # Smaller
    {"shared": "128,64", "head": 32, "name": "Small (128→64, h32)"},
    {"shared": "128,64", "head": 64, "name": "Small (128→64, h64)"},
    
    # Baseline (current)
    {"shared": "256,128", "head": 64, "name": "Baseline (256→128, h64)"},
    
    # Medium
    {"shared": "256,128", "head": 96, "name": "Medium (256→128, h96)"},
    {"shared": "192,96", "head": 64, "name": "Medium (192→96, h64)"},
    
    # Large
    {"shared": "512,256", "head": 64, "name": "Large (512→256, h64)"},
    {"shared": "256,128", "head": 128, "name": "Wide Head (256→128, h128)"},
    
    # Balanced
    {"shared": "256,256", "head": 64, "name": "Balanced (256→256, h64)"},
    {"shared": "384,192", "head": 64, "name": "Balanced (384→192, h64)"},
]

SEARCH_ROUNDS = 3  # Fast validation
SEARCH_CLIENTS = 3

def run_config(config, idx, total):
    """Run a single configuration"""
    print(f"\n[{idx}/{total}] Testing: {config['name']}")
    print(f"    Shared: {config['shared']}, Head: {config['head']}")
    
    cmd = [
        sys.executable, "main_fl_runner.py",
        "--rounds", str(SEARCH_ROUNDS),
        "--clients", str(SEARCH_CLIENTS),
        "--component", "comp4_multitask",
        "--shared", config['shared'],
        "--head-hidden", str(config['head']),
        "--head-depth", "1",
        "--dropout", "0.2"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0:
        try:
            with open("results/comp4_results/fl_results.json", "r") as f:
                data = json.load(f)
                final = data[-1]  # Last round
                
                return {
                    "Name": config['name'],
                    "Shared Layers": config['shared'],
                    "Head Width": config['head'],
                    "Global Acc": final.get('global_overall_acc', 0),
                    "Pers Acc": final.get('pers_overall_acc', 0),
                    "Gain %": final.get('gain_pct', 0),
                    "HTN AUROC": final.get('global_htn_auroc', 0),
                    "HF AUROC": final.get('global_hf_auroc', 0),
                    "Fairness Gap": final.get('fairness_gap', 0),
                    "Training Time (s)": final.get('training_time', 0),
                }
        except Exception as e:
            print(f"    Error reading results: {e}")
            return None
    else:
        print(f"    ✗ Training failed")
        return None

def main():
    print("="*80)
    print("TWO-LAYER ARCHITECTURE SEARCH")
    print("="*80)
    print(f"Testing {len(CONFIGS)} configurations × {SEARCH_ROUNDS} rounds")
    print(f"Expected time: ~{len(CONFIGS) * SEARCH_ROUNDS * 5} minutes\n")
    
    results = []
    
    for idx, config in enumerate(CONFIGS, 1):
        result = run_config(config, idx, len(CONFIGS))
        if result:
            results.append(result)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)
    
    # Sort by global accuracy (most important for FL)
    df_sorted = df.sort_values('Global Acc', ascending=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\nRanked by Global Accuracy (Best for Federated Learning):")
    print(df_sorted.to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/comp4_results/two_layer_search_{timestamp}.csv"
    df_sorted.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Find winners
    best_global = df_sorted.iloc[0]
    best_balanced = df_sorted[df_sorted['Global Acc'] > 0.55].iloc[0]
    
    print("\n" + "-"*80)
    print("🏆 WINNERS")
    print("-"*80)
    print(f"\n1️⃣  Best Global Accuracy:")
    print(f"   {best_global['Name']}")
    print(f"   Global: {best_global['Global Acc']:.4f} | Pers: {best_global['Pers Acc']:.4f}")
    print(f"   Shared: {best_global['Shared Layers']} | Head: {best_global['Head Width']}")
    
    print(f"\n2️⃣  Best Balanced (Global > 0.55):")
    if not best_balanced.empty:
        print(f"   {best_balanced['Name']}")
        print(f"   Global: {best_balanced['Global Acc']:.4f} | Pers: {best_balanced['Pers Acc']:.4f}")
        print(f"   Shared: {best_balanced['Shared Layers']} | Head: {best_balanced['Head Width']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()