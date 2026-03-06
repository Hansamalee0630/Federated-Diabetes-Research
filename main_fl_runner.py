import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
from fl_core.server import FederatedServer
from fl_core.client import FederatedClient
import os
import time
import sys
import argparse
import pickle
from sklearn.preprocessing import StandardScaler

def save_preprocessing_artifacts(X_train, scaler):
    os.makedirs("experiments/comp4_experiments", exist_ok=True)
    
    scaler_path = "experiments/comp4_experiments/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved to {scaler_path}")
    
    stats = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "X_train_shape": X_train.shape,
        "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 1
    }
    
    stats_path = "experiments/comp4_experiments/preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"   Preprocessing statistics saved to {stats_path}")

def get_model_size_mb(model):
    param_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def aggregate_metrics(metric_list):
    if not metric_list: return {}
    avg_metrics = {}

    keys = metric_list[0].keys()
    
    for k in keys:
        total = sum(d[k] for d in metric_list)
        avg_metrics[k] = total / len(metric_list)
    
    return avg_metrics

# Scalability (Test Case 8)
def run_simulation(
    num_rounds=3,
    num_clients=3,
    component_type="comp4_multitask",
    shared_layers=None,
    head_hidden=64,
    head_depth=1,
    dropout=0.2,
):
    print(f"--- Starting FL Simulation for {component_type} ---")

    try:
        sample_path = "datasets/diabetes_130/processed/client_0_X.csv"
        
        if not os.path.exists(sample_path):
             print(f"Error: Data file not found at {sample_path}")
             print("   -> Run 'python datasets/diabetes_130/preprocess.py' first!")
             return
             
        sample_data = pd.read_csv(sample_path)
        input_dim = sample_data.shape[1]
        print(f"Detected Input Features: {input_dim}")
        
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    if component_type == "comp4_multitask":
        from components.component_4.model import MultiTaskNet
        global_model = MultiTaskNet(
            input_dim=input_dim,
            shared_layers=shared_layers,
            head_hidden=head_hidden,
            head_depth=head_depth,
            dropout=dropout,
        )
        print(
            f"Loaded Multi-Task Model (Component 4) | shared_layers={shared_layers} | head_hidden={head_hidden} | head_depth={head_depth} | dropout={dropout}"
        )

    elif "singletask" in component_type:
        from components.component_4.model import SingleTaskNet
        global_model = SingleTaskNet(input_dim=input_dim)
        print(f"Loaded Single-Task Control Model for {component_type}")

    else:
        from components.component_4.model import SingleTaskNet 
        global_model = SingleTaskNet(input_dim=input_dim)
        print("Loaded Simple Model")

    server = FederatedServer(global_model)

    model_size_mb = get_model_size_mb(global_model)
    print(f"   Model Size (Communication Cost per Client): {model_size_mb:.2f} MB")

    clients = []

    for i in range(num_clients):
        data_id = i % 3
        client = FederatedClient(client_id=i, component_type=component_type)

        client.load_data(data_client_id=data_id)
        
        if hasattr(client, 'train_loader') and client.train_loader is not None and len(client.train_loader) > 0:
            clients.append(client)
        else:
            print(f"   Warning: Client {i} failed to load data. Skipping.")

    print("\n--- Training & Personalization Analysis ---")
    
    history = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== ROUND {round_num} ===")
        round_start_time = time.time()

        collected_weights = []
        
        round_global_metrics_list = []
        round_pers_metrics_list = []
        round_fairness_gap = 0

        for client in clients:
            client.set_model(copy.deepcopy(server.global_model))
            
            g_metrics, p_metrics = client.evaluate_personalization(epochs=20)
            
            round_global_metrics_list.append(g_metrics)
            round_pers_metrics_list.append(p_metrics)
            
            # 3. MEASURE FAIRNESS (Objective 2.2.iii)
            gap = client.evaluate_fairness()
            round_fairness_gap += gap

            if args.privacy:
                print(f"Client {client.client_id}: Training with DP (epsilon=2.0)...")
                client_weights = client.train_with_privacy(epochs=10, epsilon=2.0)
            else:
                client_weights = client.train(epochs=10)

            collected_weights.append(client_weights)
        
        round_duration = time.time() - round_start_time

        # AGGREGATE METRICS
        avg_global_metrics = aggregate_metrics(round_global_metrics_list)
        avg_pers_metrics = aggregate_metrics(round_pers_metrics_list)
        avg_gap = round_fairness_gap / len(clients) if clients else 0
        
        # Calculate Gain based on Overall Accuracy
        gain = 0
        if avg_global_metrics.get('overall_acc', 0) > 0:
            gain = ((avg_pers_metrics['overall_acc'] - avg_global_metrics['overall_acc']) 
                    / avg_global_metrics['overall_acc']) * 100

        print(f"   ROUND {round_num} SUMMARY:")
        print(f"   [Global] Overall Acc: {avg_global_metrics.get('overall_acc',0):.4f} | HTN AUROC: {avg_global_metrics.get('htn_auroc',0):.4f} | HF AUROC: {avg_global_metrics.get('hf_auroc',0):.4f}")
        print(f"   [Pers.]  Overall Acc: {avg_pers_metrics.get('overall_acc',0):.4f} | HTN AUROC: {avg_pers_metrics.get('htn_auroc',0):.4f} | HF AUROC: {avg_pers_metrics.get('hf_auroc',0):.4f}")
        print(f"       Gain: +{gain:.2f}% |  Fairness Gap: {avg_gap:.4f}")

        record = {
            "round": round_num,
            "fairness_gap": avg_gap,
            "gain_pct": gain,
            "training_time": round_duration
        }

        for k, v in avg_global_metrics.items(): record[f"global_{k}"] = v
        for k, v in avg_pers_metrics.items(): record[f"pers_{k}"] = v
            
        history.append(record)
        
        os.makedirs("results/comp4_results", exist_ok=True)
        with open("results/comp4_results/fl_results.json", "w") as f:
            json.dump(history, f, indent=4)

        avg_weights = server.aggregate_weights(collected_weights)
        server.update_global_model(avg_weights)
        
    os.makedirs("experiments/comp4_experiments", exist_ok=True)
    torch.save(server.global_model.state_dict(), "experiments/comp4_experiments/final_multitask_model.pth")
    print("   Model weights saved to 'experiments/comp4_experiments/final_multitask_model.pth'")

    model_config = {
        "shared_layers": shared_layers,
        "head_hidden": head_hidden,
        "head_depth": head_depth,
        "dropout": dropout
    }
    with open("experiments/comp4_experiments/model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)
    print("   Model config saved to 'experiments/comp4_experiments/model_config.json'")

    print("\n--- Saving Preprocessing Artifacts ---")
    try:
        sample_path = "datasets/diabetes_130/processed/client_0_X.csv"
        if os.path.exists(sample_path):
            X_train = pd.read_csv(sample_path).values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            save_preprocessing_artifacts(X_train, scaler)
            print(f"   Data scaling statistics: mean={scaler.mean_[:3]}, std={scaler.scale_[:3]}")
        else:
            print(f"   Warning: Could not load training data from {sample_path}")
    except Exception as e:
        print(f"   Error saving preprocessing artifacts: {e}")

    print("\n" + "="*80)
    print("   FINAL SIMULATION REPORT")
    print("="*80)
    
    final_df = pd.DataFrame(history)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    desired_cols = [
        "round",
        "global_overall_acc",
        "pers_overall_acc",
        "gain_pct",
        "global_htn_auroc",
        "pers_htn_auroc",
        "global_hf_auroc",
        "pers_hf_auroc",
        "fairness_gap",
        "training_time",
    ]


    display_cols = [col for col in desired_cols if col in final_df.columns]

    if not final_df.empty:
        print(final_df[display_cols].to_string(index=False))
        print("-" * 80)
        print(f"    Best Personalization Gain: {final_df['gain_pct'].max():.2f}%")
        print(f"    Average Fairness Gap:      {final_df['fairness_gap'].mean():.4f}")
    else:
        print("No results generated.")
        
    print("="*80)
    print("\n--- Federated Learning Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Simulation")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients (hospitals/devices)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--privacy", action="store_true", help="Enable Differential Privacy simulation")
    parser.add_argument("--shared", type=str, default="256,128", help="Comma-separated shared layer sizes (e.g., 256,128 or 256,128,64)")
    parser.add_argument("--head-hidden", type=int, default=64, help="Hidden width used inside each task head")
    parser.add_argument("--head-depth", type=int, default=1, help="Number of hidden layers inside each task head")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate applied after each shared/head layer")
    
    parser.add_argument(
        "--component",
        type=str,
        default="comp4_multitask",
        choices=["comp4_multitask", "comp4_singletask_htn", "comp4_singletask_hf"],
        help="Model type: comp4_multitask (HTN+HF+Cluster), comp4_singletask_htn (HTN only), comp4_singletask_hf (HF only)",
    )

    args = parser.parse_args()

    shared_layers = [int(x) for x in args.shared.split(",") if x.strip()]

    print(f"Launching with clients={args.clients}, rounds={args.rounds}, component={args.component}")
    run_simulation(
        num_rounds=args.rounds,
        num_clients=args.clients,
        component_type=args.component,
        shared_layers=shared_layers,
        head_hidden=args.head_hidden,
        head_depth=args.head_depth,
        dropout=args.dropout,
    )
