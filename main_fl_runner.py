import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd # Needed to check data shape
from fl_core.server import FederatedServer
from fl_core.client import FederatedClient

# --- DEFINING A SIMPLE TEST MODEL (Generic Placeholder) ---
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SimpleModel, self).__init__()
        # Dynamic input dimension!
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- MAIN SIMULATION LOOP ---
def run_simulation(num_rounds=3, num_clients=3, component_type="comp2_readmission"):
    print(f"--- Starting FL Simulation for {component_type} ---")

    # 1. AUTO-DETECT INPUT SIZE FROM DATA
    # We load one client's file just to see how many columns it has
    try:
        sample_data = pd.read_csv("datasets/diabetes_130/processed/client_0_X.csv")
        input_dim = sample_data.shape[1] # This should be 27 based on your error log
        print(f"Detected Input Features: {input_dim}")
    except FileNotFoundError:
        print("Error: Run preprocess.py first to generate data!")
        return

    # 2. Initialize Global Model based on Component Type
    if component_type == "comp4_multitask":
        # Import your custom model only if needed
        # (Make sure you create components/component_4/model.py first!)
        from components.component_4.model import MultiTaskNet
        global_model = MultiTaskNet(input_dim=input_dim)
        
    else:
        # Default for Comp 2 (Readmission)
        global_model = SimpleModel(input_dim=input_dim, output_dim=1)

    # Initialize Server
    server = FederatedServer(global_model)

    # 3. Initialize Clients
    clients = []
    for i in range(num_clients):
        client = FederatedClient(client_id=i, component_type=component_type)
        client.load_data()
        clients.append(client)

    # 4. Training Loop (Rounds)
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== ROUND {round_num} ===")
        
        collected_weights = []
        
        # A. Client Training
        for client in clients:
            # Send global weights to client
            client.set_model(copy.deepcopy(server.global_model))
            
            # Client trains locally
            client_weights = client.train(epochs=1)
            collected_weights.append(client_weights)
        
        # B. Server Aggregation
        avg_weights = server.aggregate_weights(collected_weights)
        server.update_global_model(avg_weights)
        
    print("\n--- Federated Learning Complete ---")

# if __name__ == "__main__":
#     # Test with Component 2 first to make sure the fix works
#     run_simulation(component_type="comp2_readmission")

if __name__ == "__main__":
    # NOW TESTING MY COMPONENT
    run_simulation(component_type="comp4_multitask")