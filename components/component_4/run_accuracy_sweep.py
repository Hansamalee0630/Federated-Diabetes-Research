import subprocess
import json
import pandas as pd
import os
import shutil
import sys
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# Set these to True/False based on what you need to run right now
RUN_PHASE_1_SEARCH = True   # Hyperparameter Search (12 combinations × 3 rounds)
RUN_OBJECTIVE_1 = True      # Primary Performance (10 rounds - "Golden Run")
RUN_OBJECTIVE_2 = True      # Efficiency (Multi vs Single Task)
RUN_OBJECTIVE_3 = True      # Scalability (3 vs 5 Clients)

# ==========================================
# PHASE 1: HYPERPARAMETER SEARCH SPACE
# ==========================================
SEARCH_ROUNDS = 3  # Fast check to find the winner

# The "Smart Grid" (12 Combinations)
# Head Hidden: [64, 96, 128] | Head Depth: [1, 2] | Dropout: [0.1, 0.2]
GRID_SEARCH = [
    # Group 1: 64 Neurons (Baseline Capacity)
    {"h": 64, "d": 1, "p": 0.1},
    {"h": 64, "d": 1, "p": 0.2},
    {"h": 64, "d": 2, "p": 0.1},
    {"h": 64, "d": 2, "p": 0.2},

    # Group 2: 96 Neurons (Balanced Capacity)
    {"h": 96, "d": 1, "p": 0.1},
    {"h": 96, "d": 1, "p": 0.2},
    {"h": 96, "d": 2, "p": 0.1},
    {"h": 96, "d": 2, "p": 0.2},

    # Group 3: 128 Neurons (High Capacity)
    {"h": 128, "d": 1, "p": 0.1},
    {"h": 128, "d": 1, "p": 0.2},
    {"h": 128, "d": 2, "p": 0.1},
    {"h": 128, "d": 2, "p": 0.2},
]

# Will be populated after Phase 1
BEST_PARAMS = None

# ==========================================
# EXPERIMENT DEFINITIONS
# ==========================================

# OBJ 1: The "Golden Run" for your main results table
objective1_experiments = [
    {
        "name": "Proposed MTFL (10 Rounds)",
        "component": "comp4_multitask",
        "clients": 3,
        "rounds": 10,
        "output_file": "fl_results_multitask_10rounds.json"
    }
]

# OBJ 2: Efficiency (Short runs to measure time/resource usage)
objective2_experiments = [
    {
        "name": "MultiTask (Efficiency Check)",
        "component": "comp4_multitask",
        "clients": 3,
        "rounds": 3,
        "output_file": "fl_results_multitask_3rounds.json"
    },
    {
        "name": "SingleTask (HTN Only)",
        "component": "comp4_singletask_htn",
        "clients": 3,
        "rounds": 3,
        "output_file": "fl_results_singletask_htn.json"
    },
    {
        "name": "SingleTask (HF Only)",
        "component": "comp4_singletask_hf",
        "clients": 3,
        "rounds": 3,
        "output_file": "fl_results_singletask_hf.json"
    }
]

# OBJ 3: Scalability (Does it break with more clients?)
objective3_experiments = [
    {
        "name": "Scalability (5 Clients)",
        "component": "comp4_multitask",
        "clients": 5,
        "rounds": 3, 
        "output_file": "fl_results_multitask_5clients.json"
    }
]

results = []

def run_experiment(exp, objective_name):
    """Run a single experiment and save results to a unique file"""
    print(f"\n   [{objective_name}] Testing: {exp['name']}")
    
    # Base command running from ROOT
    cmd = [
        sys.executable, "main_fl_runner.py",
        "--clients", str(exp["clients"]),
        "--rounds", str(exp["rounds"]),
        "--component", exp["component"],
        "--shared", "256,128"
    ]
    
    # Add best params ONLY for MultiTask (SingleTask models are fixed)
    if exp["component"] == "comp4_multitask":
        cmd += BEST_PARAMS
    
    print(f"   Command: {' '.join(cmd)}")
    
    # Run from current directory (Root)
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"      Experiment Failed: {exp['name']}")
        return False

    # Read result from standard output location
    try:
        source_file = "results/comp4_results/fl_results.json"
        if os.path.exists(source_file):
            with open(source_file, "r") as f:
                data = json.load(f)
                final = data[-1]
            
            # Archive the specific result file
            dest_path = f"results/comp4_results/{exp['output_file']}"
            shutil.copy(source_file, dest_path)
            print(f"      Archived to: {dest_path}")
            
            # Extract key metrics for summary
            metrics = {
                "Obj": objective_name,
                "Experiment": exp["name"],
                "Rounds": exp["rounds"],
                "Clients": exp["clients"],
                "Global Acc": f"{final.get('global_overall_acc', 0):.4f}",
                "Pers Acc": f"{final.get('pers_overall_acc', 0):.4f}",
                "HTN AUROC": f"{final.get('global_htn_auroc', 0):.4f}",
                "HF AUROC": f"{final.get('global_hf_auroc', 0):.4f}",
                "Time(s)": f"{final.get('training_time', 0):.1f}"
            }
            results.append(metrics)
            return True
        else:
            print("      Results file not found.")
            return False
    except Exception as e:
        print(f"       Error reading results: {e}")
        return False

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
print("="*80)
print("RESEARCH EXPERIMENT SUITE STARTING")
print("="*80)

# Ensure output directory exists
os.makedirs("results/comp4_results", exist_ok=True)

# ==========================================
# PHASE 1: HYPERPARAMETER SEARCH
# ==========================================
if RUN_PHASE_1_SEARCH:
    print("\n" + "="*80)
    print("PHASE 1: HYPERPARAMETER SEARCH (12 Combinations × 3 Rounds)")
    print("="*80)
    
    search_results = []
    
    for idx, params in enumerate(GRID_SEARCH, 1):
        h, d, p = params["h"], params["d"], params["p"]
        config_name = f"H{h}_D{d}_P{p}"
        
        print(f"\n[{idx}/12] Testing: Head={h}, Depth={d}, Dropout={p}")
        
        # Build command with these hyperparameters
        cmd = [
            sys.executable, "main_fl_runner.py",
            "--clients", "3",
            "--rounds", str(SEARCH_ROUNDS),
            "--component", "comp4_multitask",
            "--shared", "256,128",
            "--head-hidden", str(h),
            "--head-depth", str(d),
            "--dropout", str(p)
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            try:
                with open("results/comp4_results/fl_results.json", "r") as f:
                    data = json.load(f)
                    final = data[-1]
                    
                    pers_acc = final.get('pers_overall_acc', 0)
                    gain = final.get('gain_pct', 0)
                    
                    search_results.append({
                        "Head": h,
                        "Depth": d,
                        "Dropout": p,
                        "Config": config_name,
                        "Pers Acc": pers_acc,
                        "Gain %": gain,
                        "HTN AUROC": final.get('global_htn_auroc', 0),
                        "HF AUROC": final.get('global_hf_auroc', 0),
                        "Time(s)": final.get('training_time', 0)
                    })
                    print(f"   ✓ Pers Acc: {pers_acc:.4f}, Gain: {gain:.2f}%")
            except Exception as e:
                print(f"   ✗ Error reading results: {e}")
    
    # Find and display best configuration
    if search_results:
        search_df = pd.DataFrame(search_results)
        best_idx = search_df['Pers Acc'].idxmax()
        best_config = search_results[best_idx]
        
        print("\n" + "="*80)
        print("PHASE 1 RESULTS - TOP CONFIGURATIONS")
        print("="*80)
        print(search_df.sort_values('Pers Acc', ascending=False).to_string(index=False))
        
        print("\n" + "="*80)
        print("🏆 BEST CONFIGURATION SELECTED")
        print("="*80)
        print(f"Head Hidden: {best_config['Head']}")
        print(f"Head Depth:  {best_config['Depth']}")
        print(f"Dropout:     {best_config['Dropout']}")
        print(f"Personalized Accuracy: {best_config['Pers Acc']:.4f}")
        print(f"Personalization Gain: {best_config['Gain %']:.2f}%")
        
        # Set BEST_PARAMS for Phase 2
        BEST_PARAMS = [
            "--head-hidden", str(best_config['Head']),
            "--head-depth", str(best_config['Depth']),
            "--dropout", str(best_config['Dropout'])
        ]
        
        # Save search results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        search_csv = f"results/comp4_results/phase1_hyperparameter_search_{timestamp}.csv"
        search_df.to_csv(search_csv, index=False)
        print(f"\nSearch results saved to: {search_csv}")
    else:
        print("\n❌ Phase 1 search failed. Using default parameters.")
        BEST_PARAMS = ["--head-hidden", "96", "--head-depth", "1", "--dropout", "0.1"]

if RUN_OBJECTIVE_1:
    print("\n--- Running Objective 1 (Performance) ---")
    for exp in objective1_experiments: run_experiment(exp, "OBJ-1")

if RUN_OBJECTIVE_2:
    print("\n--- Running Objective 2 (Efficiency) ---")
    for exp in objective2_experiments: run_experiment(exp, "OBJ-2")

if RUN_OBJECTIVE_3:
    print("\n--- Running Objective 3 (Scalability) ---")
    for exp in objective3_experiments: run_experiment(exp, "OBJ-3")

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*80)
print("FINAL EXPERIMENT SUMMARY")
print("="*80)

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save Summary CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"results/comp4_results/experiment_summary_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")
else:
    print("  No results collected.")