"""
Component 4 Architecture Comparison
------------------------------------
This script compares different neural network architectures for the multitask model:
1. Baseline: 2 shared layers (256,128), head width 64
2. 3 shared layers: (256,128,64), head width 64
3. Wider heads: 2 shared layers (256,128), head width 128
4. Deeper heads: 2 shared layers (256,128), head width 64, depth 2
"""

import json
import pandas as pd
import os

def load_experiment(filename):
    """Load experiment results JSON."""
    path = f"../../results/comp4_results/{filename}"
    if not os.path.exists(path):
        print(f"Warning: {filename} not found")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def summarize_experiment(data, name):
    """Extract key metrics from the final round."""
    if not data:
        return None
    
    # Get final round (round 2)
    final_round = [r for r in data if r['round'] == 2][0]
    
    return {
        'Architecture': name,
        'Global Overall Acc': final_round['global_overall_acc'],
        'Pers Overall Acc': final_round['pers_overall_acc'],
        'Gain %': final_round['gain_pct'],
        'Global HTN AUROC': final_round['global_htn_auroc'],
        'Pers HTN AUROC': final_round['pers_htn_auroc'],
        'Global HF AUROC': final_round['global_hf_auroc'],
        'Pers HF AUROC': final_round['pers_hf_auroc'],
        'Fairness Gap': final_round['fairness_gap'],
        'Training Time (s)': final_round['training_time'],
    }

def main():
    print("="*80)
    print("COMPONENT 4: NEURAL NETWORK ARCHITECTURE SEARCH")
    print("="*80)
    print("\nPanel Question: Why 2 layers with 64 neurons? Why not 3 layers or 128 neurons?")
    print("\nExperiments conducted:")
    print("  1. Baseline:      2 shared layers [256,128], head width 64, depth 1")
    print("  2. 3-Layer Body:  3 shared layers [256,128,64], head width 64, depth 1")
    print("  3. Wider Heads:   2 shared layers [256,128], head width 128, depth 1")
    print("  4. Deeper Heads:  2 shared layers [256,128], head width 64, depth 2")
    print("\n" + "-"*80)
    
    # Load all experiments
    experiments = [
        ('fl_results_baseline_256x128_head64.json', 'Baseline (2-layer, head=64)'),
        ('fl_results_shared_256x128x64_head64.json', '3-Layer Shared (256,128,64)'),
        ('fl_results_shared_256x128_head128.json', 'Wider Heads (head=128)'),
        ('fl_results_shared_256x128_head64x2.json', 'Deeper Heads (head=64x2)'),
    ]
    
    results = []
    for filename, name in experiments:
        data = load_experiment(filename)
        summary = summarize_experiment(data, name)
        if summary:
            results.append(summary)
    
    # Create comparison dataframe
    df = pd.DataFrame(results)
    
    # Display with formatting
    print("\n### FINAL ROUND (Round 2) COMPARISON ###\n")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("\n### KEY FINDINGS ###\n")
    
    # Find best by different metrics
    best_global_acc = df.loc[df['Global Overall Acc'].idxmax()]
    best_pers_acc = df.loc[df['Pers Overall Acc'].idxmax()]
    best_htn = df.loc[df['Pers HTN AUROC'].idxmax()]
    best_hf = df.loc[df['Pers HF AUROC'].idxmax()]
    best_fairness = df.loc[df['Fairness Gap'].idxmin()]
    
    print(f"1. Best Global Accuracy:        {best_global_acc['Architecture']} ({best_global_acc['Global Overall Acc']:.4f})")
    print(f"2. Best Personalized Accuracy:  {best_pers_acc['Architecture']} ({best_pers_acc['Pers Overall Acc']:.4f})")
    print(f"3. Best HTN AUROC (Personalized): {best_htn['Architecture']} ({best_htn['Pers HTN AUROC']:.4f})")
    print(f"4. Best HF AUROC (Personalized):  {best_hf['Architecture']} ({best_hf['Pers HF AUROC']:.4f})")
    print(f"5. Best Fairness (Lowest Gap):   {best_fairness['Architecture']} ({best_fairness['Fairness Gap']:.4f})")
    
    print("\n" + "-"*80)
    print("="*80)
    
    # Save comparison
    output_path = "../../results/comp4_results/architecture_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\nComparison table saved to: {output_path}")

if __name__ == "__main__":
    main()
