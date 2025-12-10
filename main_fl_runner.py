import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
from fl_core.server import FederatedServer
from fl_core.client import FederatedClient
import os

# --- 1. DEFINE SIMPLE MODEL (For Team Member 2) ---
# We keep this here so the code works out-of-the-box for the Causal Component
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1) # Single output (Readmission)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- 2. MAIN SIMULATION LOOP ---
# Scalability (Test Case 8) -> You can easily verify this right now. Open main_fl_runner.py and change the default to num_clients=10.
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
        # Import your Custom Multi-Task Model
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

    # C. INITIALIZE CLIENTS
    clients = []
    for i in range(num_clients):
        client = FederatedClient(client_id=i, component_type=component_type)
        client.load_data()
        clients.append(client)

    # D. TRAINING LOOP (With Personalization Tracking)
    print("\n--- Training & Personalization Analysis ---")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== ROUND {round_num} ===")
        collected_weights = []
        
        round_global_acc = 0
        round_pers_acc = 0
        
        for client in clients:
            # 1. Update Client with Global Model
            client.set_model(copy.deepcopy(server.global_model))
            
            # 2. MEASURE GAIN (Personalization Step)
            # Run before standard training to track improvement
            g_acc, p_acc = client.evaluate_personalization(epochs=10)
            round_global_acc += g_acc
            round_pers_acc += p_acc
            
            # 3. MEASURE FAIRNESS (Objective 2.2.iii)
            # We run this check to see if the model is biased
            client.evaluate_fairness()
            
            # 4. Standard FL Training
            client_weights = client.train(epochs=1)
            collected_weights.append(client_weights)
        
        # Calculate Averages
        avg_global = round_global_acc / num_clients
        avg_pers = round_pers_acc / num_clients
        gain = (avg_pers - avg_global) * 100
        
        print(f"📊 ROUND {round_num} METRICS:")
        print(f"   Global Model Accuracy: {avg_global:.4f}")
        print(f"   Personalized Accuracy: {avg_pers:.4f}")
        print(f"   🚀 Personalization Gain: +{gain:.2f}%")
        
        # 4. Aggregation
        avg_weights = server.aggregate_weights(collected_weights)
        server.update_global_model(avg_weights)
        
    print("\n--- Federated Learning Complete ---")

if __name__ == "__main__":
    # You can change this to "comp2_readmission" to test Member 2's code
    run_simulation(component_type="comp4_multitask")

# if __name__ == "__main__":
#     run_simulation(component_type="comp4_singletask")