import torch
import torch.nn as nn
import torch.nn.functional as F

class NephropathyNet(nn.Module):
    def __init__(self, input_size):
        super(NephropathyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        # REMOVED self.sigmoid

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # REMOVED sigmoid here. We return raw logits now.
        x = self.fc3(x) 
        return x

class CVDNet(nn.Module):
    def __init__(self, input_size):
        super(CVDNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        # REMOVED self.sigmoid

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # REMOVED sigmoid here.
        x = self.fc3(x)
        return x