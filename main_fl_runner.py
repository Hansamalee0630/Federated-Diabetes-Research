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

# --- 1. DEFINE SIMPLE MODEL (For Team Member 2) ---
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1) # Single output (Readmission)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def get_model_size_mb(model):
    """Calculates the size of the model parameters in Megabytes"""
    param_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

# --- 2. MAIN SIMULATION LOOP ---
# Scalability (Test Case 8) -> Verify by opening main_fl_runner.py and change the default to num_clients=10.
def run_simulation(num_rounds=3, num_clients=3, component_type="comp4_multitask"):
    print(f"--- Starting FL Simulation for {component_type} ---")

    # A. AUTO-DETECT INPUT SIZE
    try:
        # We try to load Client 0's data to check the shape 
        sample_path = "datasets/diabetes_130/processed/client_0_X.csv"
        
        # Fallback check if data exists
        if not os.path.exists(sample_path):
             print(f"❌ Error: Data file not found at {sample_path}")
             print("   -> Run 'python datasets/diabetes_130/preprocess.py' first!")
             return
             
        sample_data = pd.read_csv(sample_path)
        input_dim = sample_data.shape[1]
        print(f"Detected Input Features: {input_dim}")
        
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # B. INITIALIZE GLOBAL MODEL
    if component_type == "comp4_multitask":
        from components.component_4.model import MultiTaskNet
        global_model = MultiTaskNet(input_dim=input_dim)
        print("Loaded Multi-Task Model (Component 4)")

    elif component_type == "comp4_singletask":
        from components.component_4.model import SingleTaskNet
        global_model = SingleTaskNet(input_dim=input_dim)
        print("Loaded Single-Task Control Model (Component 4)")
    
    # # === MEMBER 1 ADDS THIS BLOCK ===
    # elif component_type == "comp1_multimodal":
    #     from components.component_1.model import Model # He creates this file
    #     global_model = Model() 
    #     print("Loaded Complication Model (Component 1)")

    # # === MEMBER 3 ADDS THIS BLOCK ===
    # elif component_type == "comp3_multimodal":
    #     from components.component_3.model import CNNModel # He creates this file
    #     global_model = CNNModel() 
    #     print("Loaded CNN Model (Component 3)")
    

    else:
        # Use the SimpleModel defined above (For Component 2)
        global_model = SimpleModel(input_dim)
        print("Loaded Simple Model (Component 2)")

    # Initialize Server
    server = FederatedServer(global_model)

    # Calculate Communication Cost (Model Size)
    model_size_mb = get_model_size_mb(global_model)
    print(f"📦 Model Size (Communication Cost per Client): {model_size_mb:.2f} MB")

    # C. INITIALIZE CLIENTS
    clients = []
    # Note: We cycle through available data partitions if num_clients > 3
    # This simulates having more clients by reusing the 3 datasets we prepared

    for i in range(num_clients):
        data_id = i % 3  # Reuse dataset 0, 1, 2 if we ask for 10 clients
        client = FederatedClient(client_id=i, component_type=component_type)
        # We override the default data loading to allow reuse
        # (Assuming FederatedClient code handles loading "client_{data_id}_X.csv")
        # For now, we rely on the client class default behavior. 
        # If the client class hardcodes file loading, ensure it handles i > 2 correctly.
        # Simple fix: Update client.py or preprocess more data. 
        # For this simulation, reusing partitions is acceptable for scalability testing.
        
        # --- THE FIX IS HERE ---
        # We explicitly pass 'data_id' to load_data so it knows which file to read.
        client.load_data(data_client_id=data_id)
        
        # Only add client if data loaded successfully
        if hasattr(client, 'train_loader') and client.train_loader is not None and len(client.train_loader) > 0:
            clients.append(client)
        else:
            print(f"⚠️ Warning: Client {i} failed to load data. Skipping.")

    # D. TRAINING LOOP (With Personalization Tracking)
    print("\n--- Training & Personalization Analysis ---")
    
    # === NEW: LOGGING LIST ===
    history = []
    total_training_time = 0

    for round_num in range(1, num_rounds + 1):
        print(f"\n=== ROUND {round_num} ===")
        
        # Start Timer
        round_start_time = time.time()

        collected_weights = []
        round_global_acc = 0
        round_pers_acc = 0
        round_fairness_gap = 0

        
        for client in clients:
            # 1. Update Client with Global Model
            client.set_model(copy.deepcopy(server.global_model))
            
            # 2. MEASURE GAIN (Personalization Step)
            # Run before standard training to track improvement
            # g_acc, p_acc = client.evaluate_personalization(epochs=10)
            g_acc, p_acc = client.evaluate_personalization(epochs=1) # Reduced epochs for speed in testing
            round_global_acc += g_acc
            round_pers_acc += p_acc
            
            # 3. MEASURE FAIRNESS (Objective 2.2.iii)
            # We run this check to see if the model is biased
            gap = client.evaluate_fairness()
            round_fairness_gap += gap

            # 4. Standard FL Training
            client_weights = client.train(epochs=1)
            collected_weights.append(client_weights)
        
        # Stop Timer
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        total_training_time += round_duration

        # Calculate Averages
        if len(clients) > 0:
            avg_global = round_global_acc / len(clients)
            avg_pers = round_pers_acc / len(clients)
            avg_gap = round_fairness_gap / len(clients)
        else:
            avg_global = avg_pers = avg_gap = 0

        gain = (avg_pers - avg_global) * 100
        
        # Estimate Total Data Transferred (Upload + Download for all clients)
        # Round Data = (Model Down + Model Up) * Num Clients
        round_comm_cost = (model_size_mb * 2) * num_clients

        print(f"\n📊 ROUND {round_num} METRICS:")
        print(f"   ⏱️ Time Taken: {round_duration:.2f}s")
        print(f"   📡 Est. Comm Cost: {round_comm_cost:.2f} MB")
        print(f"   🚀 Personalization Gain: +{gain:.2f}%")

        # SAVE TO HISTORY
        history.append({
            "round": round_num,
            "global_accuracy": avg_global,
            "personalized_accuracy": avg_pers,
            "personalization_gain": gain,
            "fairness_gap": avg_gap,
            "training_time_sec": round_duration,    
            "comm_cost_mb": round_comm_cost          
        })
        
        # Dump to JSON
        os.makedirs("results/comp4_results", exist_ok=True)
        with open("results/comp4_results/fl_results.json", "w") as f:
            json.dump(history, f)

        # 4. Aggregation
        avg_weights = server.aggregate_weights(collected_weights)
        server.update_global_model(avg_weights)
        
    # Save Final Model
    os.makedirs("experiments/comp4_experiments", exist_ok=True)
    torch.save(server.global_model.state_dict(), "experiments/comp4_experiments/final_multitask_model.pth")
    print("💾 Model weights saved to 'experiments/comp4_experiments/final_multitask_model.pth'")

    # === E. FINAL SUMMARY REPORT (Best Practice) ===
    print("\n" + "="*80)
    print("🏁 FINAL SIMULATION REPORT")
    print("="*80)
    
    # Convert history to DataFrame for pretty printing
    final_df = pd.DataFrame(history)
    
    # Select columns to display (include timing and comms for scalability reporting)
    display_cols = [
        "round",
        "global_accuracy",
        "personalized_accuracy",
        "personalization_gain",
        "fairness_gap",
        "training_time_sec",
        "comm_cost_mb",
    ]
    
    # Print the table without the index (cleaner look)
    if not final_df.empty:
        print(final_df[display_cols].to_string(index=False))
        print("-" * 80)
        print(f"🏆 Best Personalization Gain: {final_df['personalization_gain'].max():.2f}%")
        print(f"⚖️  Average Fairness Gap:      {final_df['fairness_gap'].mean():.4f}")
    else:
        print("No results generated.")
        
    print("="*80)
    print("\n--- Federated Learning Complete ---")


if __name__ == "__main__":
    # Simple CLI to support scalability testing without extra scripts
    parser = argparse.ArgumentParser(description="Run Federated Learning Simulation")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients (hospitals/devices)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument(
        "--component",
        type=str,
        default="comp4_multitask",
        choices=["comp4_multitask", "comp4_singletask", "comp2_readmission"],
        help="Which component/model to run",
    )

    args = parser.parse_args()

    print(f"Launching with clients={args.clients}, rounds={args.rounds}, component={args.component}")
    run_simulation(num_rounds=args.rounds, num_clients=args.clients, component_type=args.component)

# if __name__ == "__main__":
#     run_simulation(component_type="comp4_singletask")