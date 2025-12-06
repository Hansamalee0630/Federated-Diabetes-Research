import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim=95): 
        # input_dim=95 because One-Hot Encoding increased the feature count
        # If your actual feature count differs, the code will auto-detect it later.
        super(MultiTaskNet, self).__init__()
        
        # --- SHARED LAYERS (The "Body") ---
        # These layers learn general patterns common to both diseases
        self.shared_fc1 = nn.Linear(input_dim, 64)
        self.shared_fc2 = nn.Linear(64, 32)
        
        # --- TASK SPECIFIC HEADS (The "Limbs") ---
        
        # Head 1: Hypertension Prediction (Binary: Yes/No)
        self.head_hypertension = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Outputs probability between 0 and 1
        )
        
        # Head 2: Heart Failure Prediction (Binary: Yes/No)
        self.head_heart_failure = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Outputs probability between 0 and 1
        )

    def forward(self, x):
        # Pass through shared layers
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        # Split into two tasks
        out_htn = self.head_hypertension(x)
        out_hf = self.head_heart_failure(x)
        
        # Return both predictions
        return out_htn, out_hf