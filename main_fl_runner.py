from copy import copy
import torch
from fl_core.server import FederatedServer
from fl_core.client import FederatedClient
# Import your specific model here later (e.g., from models.comp4_model import MultiTaskNet)
import torch.nn as nn
import torch.nn.functional as F

# --- DEFINING A SIMPLE TEST MODEL (Placeholder) ---
# Everyone will replace this with their own component's model!
class SimpleModel(nn.Module):
    def __init__(self, input_dim=12, output_dim=1): # 12 features in diabetes dataset
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- MAIN SIMULATION LOOP ---
def run_simulation(num_rounds=3, num_clients=3, component_type="comp2_readmission"):
    print(f"--- Starting FL Simulation for {component_type} ---")

    # 1. Initialize Global Model & Server
    # Note: Component 4 will change output_dim to 2 (for Hypertension + Heart Failure)
    global_model = SimpleModel(output_dim=1) 
    server = FederatedServer(global_model)

    # 2. Initialize Clients
    clients = []
    for i in range(num_clients):
        client = FederatedClient(client_id=i, component_type=component_type)
        client.load_data()
        clients.append(client)

    # 3. Training Loop (Rounds)
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

if __name__ == "__main__":
    # You can change this to 'comp4_multitask' to test your part
    run_simulation(component_type="comp2_readmission")