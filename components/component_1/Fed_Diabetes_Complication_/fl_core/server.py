import torch
import copy

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate_weights(self, client_weights_list):
        avg_weights = copy.deepcopy(client_weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(client_weights_list)):
                avg_weights[key] += client_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(client_weights_list))
        
        self.global_model.load_state_dict(avg_weights)
        return self.global_model