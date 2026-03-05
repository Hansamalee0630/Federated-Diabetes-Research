import json
import matplotlib.pyplot as plt
import pandas as pd

# Load the Golden Run data
file_path = "results/comp4_results/fl_results_multitask_10rounds.json"

try:
    with open(file_path, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Setup the figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Accuracy (Left Y-Axis)
    rounds = df["round"]
    ax1.plot(rounds, df["global_overall_acc"], marker="o", linestyle="-", linewidth=2.5, label="Global Acc", color="#1f77b4") # Blue
    ax1.plot(rounds, df["pers_overall_acc"], marker="s", linestyle="--", linewidth=2.5, label="Pers. Acc", color="#ff7f0e")   # Orange
    
    ax1.set_xlabel("Communication Rounds", fontsize=12)
    ax1.set_ylabel("Accuracy (0.0 - 1.0)", fontsize=12)
    ax1.set_ylim(0.4, 0.8) # Adjusted for your 65% range
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Create Right Y-Axis for Gain %
    ax2 = ax1.twinx()
    ax2.plot(rounds, df["gain_pct"], marker="^", linestyle="-.", linewidth=2, label="Pers. Gain (%)", color="#2ca02c") # Green
    ax2.set_ylabel("Personalization Gain (%)", fontsize=12, color="#2ca02c")
    ax2.tick_params(axis='y', labelcolor="#2ca02c")
    ax2.set_ylim(0, 35) # Adjusted for your 30% gain

    # Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11, shadow=True)

    plt.title("Convergence Analysis: Global vs. Personalized Performance", fontsize=14)
    plt.tight_layout()
    
    # Save
    output_path = "results/comp4_results/convergence_final.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to {output_path}")
    plt.show()

except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure you ran the 'run_accuracy_sweep.py' script first!")