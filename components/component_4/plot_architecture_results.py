import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
results_dir = "results/comp4_results"
files = {
    "Baseline (2-Layer Shared)": "fl_results_baseline_256x128_head64.json",
    "Deep Shared (3-Layer)": "fl_results_shared_256x128x64_head64.json",
    "Wide Heads (128 units)": "fl_results_shared_256x128_head128.json",
    "Deep Heads (2-Layer)": "fl_results_shared_256x128_head64x2.json"
}

data = []

print("Analyzing Architecture Results...")

for label, filename in files.items():
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            history = json.load(f)
            # Get the metrics from the final round (Round 2)
            final_round = history[-1]
            data.append({
                "Architecture": label,
                "Global Accuracy": final_round["global_overall_acc"],
                "Personalized Accuracy": final_round["pers_overall_acc"],
                "Fairness Gap": final_round["fairness_gap"]
            })
    else:
        print(f"Warning: {filename} not found.")

df = pd.DataFrame(data)

# --- PLOTTING ---
sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar Chart for Accuracy
x = range(len(df))
width = 0.35

bars1 = ax1.bar([i - width/2 for i in x], df["Global Accuracy"], width, label="Global Acc", color="#4c72b0")
bars2 = ax1.bar([i + width/2 for i in x], df["Personalized Accuracy"], width, label="Personalized Acc", color="#55a868")

ax1.set_ylabel("Accuracy", fontsize=12)
ax1.set_title("Neural Network Architecture Search (Diabetes 130-US)", fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df["Architecture"], rotation=15)
ax1.set_ylim(0.4, 0.7) # Zoom in to show differences
ax1.legend(loc="upper left")

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)

# Secondary Axis for Fairness Gap (Line Chart)
ax2 = ax1.twinx()
ax2.plot(x, df["Fairness Gap"], color="#c44e52", marker="o", linestyle="--", linewidth=2, label="Fairness Gap (Lower is Better)")
ax2.set_ylabel("Fairness Gap", color="#c44e52", fontsize=12)
ax2.tick_params(axis='y', labelcolor="#c44e52")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("architecture_search_proof.png", dpi=300)
print("\nChart saved as 'architecture_search_proof.png'. Add this to your presentation!")
print("\n--- SUMMARY TABLE ---")
print(df)