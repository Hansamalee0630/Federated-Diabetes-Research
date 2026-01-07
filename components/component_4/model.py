import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim): 
        super(MultiTaskNet, self).__init__()
        
        # --- SHARED BODY (Feature Extractor) ---
        # Increased size and added Batch Norm for stability
        self.shared_fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.shared_fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        # --- TASK HEADS ---
        # Head 1: Hypertension
        self.head_htn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Heart Failure
        self.head_hf = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # --- HEAD 3: Comorbidity Cluster (Multi-Class: 3 Classes) ---
        # 0=Metabolic, 1=Circulatory, 2=Complex
        self.head_cluster = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3) # Output size 3 (Logits)
        )

    def forward(self, x):
        # Body
        x = F.relu(self.bn1(self.shared_fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.shared_fc2(x)))
        x = self.dropout2(x)
        
        # Heads
        return self.head_htn(x), self.head_hf(x), self.head_cluster(x)


# MTFL vs. Single-Task (Objective 2.2.ii)
class SingleTaskNet(nn.Module):
    def __init__(self, input_dim): 
        super(SingleTaskNet, self).__init__()
        
        # Similar body to MultiTask, but simpler
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        
        # ONLY ONE HEAD (Single Output)
        self.head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        
        return self.head(x)