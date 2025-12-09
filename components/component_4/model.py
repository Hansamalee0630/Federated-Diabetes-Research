# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiTaskNet(nn.Module):
#     def __init__(self, input_dim=95): 
#         # input_dim=95 because One-Hot Encoding increased the feature count
#         # If your actual feature count differs, the code will auto-detect it later.
#         super(MultiTaskNet, self).__init__()
        
#         # --- SHARED LAYERS (The "Body") ---
#         # These layers learn general patterns common to both diseases
#         self.shared_fc1 = nn.Linear(input_dim, 64)
#         self.shared_fc2 = nn.Linear(64, 32)
        
#         # --- TASK SPECIFIC HEADS (The "Limbs") ---
        
#         # Head 1: Hypertension Prediction (Binary: Yes/No)
#         self.head_hypertension = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid() # Outputs probability between 0 and 1
#         )
        
#         # Head 2: Heart Failure Prediction (Binary: Yes/No)
#         self.head_heart_failure = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1),
#             nn.Sigmoid() # Outputs probability between 0 and 1
#         )

#     def forward(self, x):
#         # Pass through shared layers
#         x = F.relu(self.shared_fc1(x))
#         x = F.relu(self.shared_fc2(x))
        
#         # Split into two tasks
#         out_htn = self.head_hypertension(x)
#         out_hf = self.head_heart_failure(x)
        
#         # Return both predictions
#         return out_htn, out_hf

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim): 
        super(MultiTaskNet, self).__init__()
        
        # --- SHARED BODY (Feature Extractor) ---
        # Increased size and added Batch Norm for stability
        self.shared_fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3) # Prevents overfitting
        
        self.shared_fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        # --- TASK HEADS ---
        # Head 1: Hypertension
        self.head_htn = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Heart Failure
        self.head_hf = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Body
        x = F.relu(self.bn1(self.shared_fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.shared_fc2(x)))
        x = self.dropout2(x)
        
        # Heads
        return self.head_htn(x), self.head_hf(x)


# MTFL vs. Single-Task (Objective 2.2.ii)
class SingleTaskNet(nn.Module):
    def __init__(self, input_dim): 
        super(SingleTaskNet, self).__init__()
        
        # Similar body to MultiTask, but simpler
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        
        # ONLY ONE HEAD (Single Output)
        self.head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        return self.head(x)