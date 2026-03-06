import torch
import copy

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate_weights(self, client_weights):
        """
        Federated Averaging (FedAvg).
        Takes a list of client state_dicts and averages them.
        """
        avg_weights = copy.deepcopy(client_weights[0])
        
        for key in avg_weights.keys():
            layer_stack = torch.stack([w[key].float() for w in client_weights], dim=0)
            
            avg_weights[key] = torch.mean(layer_stack, dim=0)
            
        return avg_weights

    def update_global_model(self, avg_weights):
        """Updates the server's model with the new averaged weights."""
        self.global_model.load_state_dict(avg_weights)
        print("Server: Global model updated.")