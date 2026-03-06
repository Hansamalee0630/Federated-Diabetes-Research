import torch
import copy
import numpy as np

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate_weights(self, client_weights):
        """
        Federated Averaging (FedAvg).
        Takes a list of client state_dicts and averages them.
        """
        # Initialize an average dictionary with the same keys as the model
        avg_weights = copy.deepcopy(client_weights[0])
        
        # Iterate through every layer in the weight dictionary
        for key in avg_weights.keys():
            # Stack the weights from all clients for this layer
            # shape: (num_clients, layer_shape...)
            layer_stack = torch.stack([w[key].float() for w in client_weights], dim=0)
            
            # Calculate mean
            avg_weights[key] = torch.mean(layer_stack, dim=0)
            
        return avg_weights

    def update_global_model(self, avg_weights):
        """Updates the server's model with the new averaged weights."""
        self.global_model.load_state_dict(avg_weights)
        print("Server: Global model updated.")