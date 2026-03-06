import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNet(nn.Module):
    def __init__(
        self,
        input_dim,
        shared_layers=None,
        head_hidden=64,
        head_depth=1,
        dropout=0.2,
    ):
        super(MultiTaskNet, self).__init__()

        shared_layers = shared_layers or [256, 128]
        self.dropout_rate = dropout

        # SHARED BODY (Feature Extractor)
        shared_dims = [input_dim] + shared_layers
        self.shared_fcs = nn.ModuleList(
            [nn.Linear(shared_dims[i], shared_dims[i + 1]) for i in range(len(shared_layers))]
        )
        self.shared_bns = nn.ModuleList(
            [nn.BatchNorm1d(shared_dims[i + 1]) for i in range(len(shared_layers))]
        )
        self.shared_dropout = nn.Dropout(self.dropout_rate)

        # TASK HEADS
        # HTN and HF heads output raw logits (no sigmoid) for use with BCEWithLogitsLoss
        last_shared_dim = shared_layers[-1]
        self.head_htn = self._build_head(last_shared_dim, out_dim=1, head_hidden=head_hidden, head_depth=head_depth, activation=None)
        self.head_hf = self._build_head(last_shared_dim, out_dim=1, head_hidden=head_hidden, head_depth=head_depth, activation=None)
        self.head_cluster = self._build_head(last_shared_dim, out_dim=3, head_hidden=head_hidden, head_depth=head_depth, activation=None)

    def _build_head(self, in_dim, out_dim, head_hidden, head_depth, activation):
        layers = []
        for _ in range(head_depth):
            layers.append(nn.Linear(in_dim, head_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            in_dim = head_hidden
        layers.append(nn.Linear(in_dim, out_dim))
        if activation:
            layers.append(activation)
        return nn.Sequential(*layers)

        # --- HEAD 3: Comorbidity Cluster (Multi-Class: 3 Classes) ---
        # 0=Metabolic, 1=Circulatory, 2=Complex
        self.head_cluster = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3) # Output size 3 (Logits)
        )

    def forward(self, x):
        for fc, bn in zip(self.shared_fcs, self.shared_bns):
            x = F.relu(bn(fc(x)))
            x = self.shared_dropout(x)
        return self.head_htn(x), self.head_hf(x), self.head_cluster(x)


# MTFL vs. Single-Task
class SingleTaskNet(nn.Module):
    def __init__(self, input_dim): 
        super(SingleTaskNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        
        # ONLY ONE HEAD (Single Output)
        self.head = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        
        return self.head(x)