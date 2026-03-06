import torch
import torch.nn as nn
from torchvision import models

# --- ARCHITECTURE 1: EHR MLP ---
# Matched to her training results: 8 features -> 128 -> 64 -> 32
class EHRClassifier(nn.Module):
    def __init__(self, n_features=8, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(EHRClassifier, self).__init__()
        layers = []
        input_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim), # Her model uses BN before ReLU
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classifier must be Sequential to match the 'classifier.0.weight' key
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# --- ARCHITECTURE 2: Retinal EfficientNet-B3 ---
class BinaryDRClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b3', dropout=0.3):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        n = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() 
        self.feature_dim = n  # 1536 for B3
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f)

# --- ARCHITECTURE 3: Global Multimodal Fusion ---
class GlobalMultimodalModel(nn.Module):
    def __init__(self, ehr_encoder, ret_encoder, ehr_dim=32, img_dim=1536):
        super().__init__()
        self.ehr_encoder = ehr_encoder
        self.ret_encoder = ret_encoder
        
        # UPDATED: Match the saved model's 8-layer architecture
        self.classifier = nn.Sequential(
            nn.Linear(ehr_dim + img_dim, 256),      # Layer 0
            nn.ReLU(),                               # Layer 1
            nn.BatchNorm1d(256),                     # Layer 2
            nn.Dropout(0.3),                         # Layer 3
            nn.Linear(256, 64),                      # Layer 4
            nn.ReLU(),                               # Layer 5
            nn.Dropout(0.2),                         # Layer 6 (ADDED - was missing)
            nn.Linear(64, 1)                         # Layer 7
        )

    def forward(self, x_ehr, x_img):
        f_ehr = self.ehr_encoder(x_ehr)
        f_img = self.ret_encoder(x_img)
        f_combined = torch.cat((f_ehr, f_img), dim=1)
        return self.classifier(f_combined)    




# import torch
# import torch.nn as nn
# from torchvision import models

# # --- ARCHITECTURE 1: EHR MLP ---
# # Matched to her checkpoint: 8 inputs -> 128 -> 64 -> 32 -> Output
# class EHRClassifier(nn.Module):
#     def __init__(self, n_features=8, hidden_dims=[128, 64, 32], dropout_rate=0.3):
#         super(EHRClassifier, self).__init__()
#         layers = []
#         input_dim = n_features
        
#         for h_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(input_dim, h_dim),
#                 nn.BatchNorm1d(h_dim), # BN before ReLU as per her training
#                 nn.ReLU(),
#                 nn.Dropout(dropout_rate)
#             ])
#             input_dim = h_dim
            
#         self.feature_extractor = nn.Sequential(*layers)
        
#         # Classifier must be Sequential to match "classifier.0.weight" in checkpoint
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 1)
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         return self.classifier(features)

# # --- ARCHITECTURE 2: Retinal EfficientNet-B3 ---
# class BinaryDRClassifier(nn.Module):
#     def __init__(self, model_name='efficientnet_b3', dropout=0.3):
#         super().__init__()
#         self.backbone = models.efficientnet_b3(weights=None)
#         n = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Identity() 
#         self.feature_dim = n  # 1536 for B3
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(n, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(dropout * 0.5),
#             nn.Linear(256, 1)
#         )

#     def forward(self, x):
#         f = self.backbone(x)
#         return self.classifier(f)

# # --- ARCHITECTURE 3: Global Multimodal Fusion ---
# class GlobalMultimodalModel(nn.Module):
#     def __init__(self, ehr_encoder, ret_encoder, ehr_dim=32, img_dim=1536):
#         super().__init__()
#         self.ehr_encoder = ehr_encoder
#         self.ret_encoder = ret_encoder
        
#         # Head that takes the concatenated 1568-dim vector
#         self.classifier = nn.Sequential(
#             nn.Linear(ehr_dim + img_dim, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x_ehr, x_img):
#         # Extract features
#         f_ehr = self.ehr_encoder(x_ehr)
#         f_img = self.ret_encoder(x_img)
        
#         # Fusion via Concatenation
#         f_combined = torch.cat((f_ehr, f_img), dim=1)
#         return self.classifier(f_combined)

# import torch
# import torch.nn as nn
# from torchvision import models

# # --- ARCHITECTURE 1: EHR MLP ---
# class EHRClassifier(nn.Module):
#     def __init__(self, n_features=8, hidden_dims=[64, 32], dropout_rate=0.3):
#         super(EHRClassifier, self).__init__()
#         layers = []
#         input_dim = n_features
#         for h_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(input_dim, h_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(h_dim),
#                 nn.Dropout(dropout_rate)
#             ])
#             input_dim = h_dim
#         self.feature_extractor = nn.Sequential(*layers)
#         self.classifier = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         return self.classifier(features)

# # --- ARCHITECTURE 2: Retinal EfficientNet-B3 ---
# class BinaryDRClassifier(nn.Module):
#     def __init__(self, model_name='efficientnet_b3', dropout=0.3):
#         super().__init__()
#         # Use newer weights parameter for compatibility
#         self.backbone = models.efficientnet_b3(weights=None)
#         n = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Identity() # Removes original top layer
#         self.feature_dim = n  # 1536 for B3
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(n, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(dropout * 0.5),
#             nn.Linear(256, 1)
#         )

#     def forward(self, x):
#         f = self.backbone(x)
#         return self.classifier(f)

# # --- ARCHITECTURE 3: Global Multimodal Fusion ---
# class GlobalMultimodalModel(nn.Module):
#     def __init__(self, ehr_encoder, ret_encoder, ehr_dim=32, img_dim=1536):
#         super().__init__()
#         self.ehr_encoder = ehr_encoder
#         self.ret_encoder = ret_encoder
#         self.classifier = nn.Sequential(
#             nn.Linear(ehr_dim + img_dim, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x_ehr, x_img):
#         f_ehr = self.ehr_encoder(x_ehr)
#         f_img = self.ret_encoder(x_img)
#         f_combined = torch.cat((f_ehr, f_img), dim=1)
#         return self.classifier(f_combined)


# import gdown
# import torch
# import torch.nn as nn
# from torchvision import models

# # --- ARCHITECTURE 1: EHR MLP ---
# class EHRClassifier(nn.Module):
#     def __init__(self, n_features=8, hidden_dims=[64, 32], dropout_rate=0.3):
#         super(EHRClassifier, self).__init__()
#         layers = []
#         input_dim = n_features
#         for h_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(input_dim, h_dim),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(h_dim),
#                 nn.Dropout(dropout_rate)
#             ])
#             input_dim = h_dim
#         self.feature_extractor = nn.Sequential(*layers)
#         self.classifier = nn.Linear(input_dim, 1)

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         return self.classifier(features)

# # --- ARCHITECTURE 2: Retinal EfficientNet-B3 ---
# class BinaryDRClassifier(nn.Module):
#     def __init__(self, model_name='efficientnet_b3', dropout=0.3):
#         super().__init__()
#         self.backbone = models.efficientnet_b3(weights=None)
#         n = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Identity()
#         self.feature_dim = n  # 1536 for B3
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(n, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(dropout * 0.5),
#             nn.Linear(256, 1)
#         )

#     def forward(self, x):
#         f = self.backbone(x)
#         return self.classifier(f)

# # --- ARCHITECTURE 3: Global Multimodal Fusion ---
# class GlobalMultimodalModel(nn.Module):
#     def __init__(self, ehr_encoder, ret_encoder, ehr_dim=32, img_dim=1536):
#         super().__init__()
#         self.ehr_encoder = ehr_encoder
#         self.ret_encoder = ret_encoder
#         self.classifier = nn.Sequential(
#             nn.Linear(ehr_dim + img_dim, 256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.3),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x_ehr, x_img):
#         f_ehr = self.ehr_encoder(x_ehr)
#         f_img = self.ret_encoder(x_img)
#         f_combined = torch.cat((f_ehr, f_img), dim=1)
#         return self.classifier(f_combined)
    
    
    
    
# """
# Model architectures for Multimodal Fusion.

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   CRITICAL: Each class MUST match your Colab training code
#   EXACTLY — same layer names, same dimensions, same order.
#   Otherwise load_state_dict() will throw a KeyError.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# """

# import torch
# import torch.nn as nn


# # ================================================================
# # MODEL 1: EHR MLP
# # ================================================================
# class EHR_MLP(nn.Module):
#     """
#     PLACEHOLDER — Replace internals with YOUR exact Colab architecture.
#     Keep the class name, just change the layers inside.
    
#     To find your architecture, run this in Colab:
#         print(model)
#     and replicate every layer here.
#     """
#     def __init__(self, input_dim=128, hidden_dims=None, num_classes=2):
#         super(EHR_MLP, self).__init__()
#         if hidden_dims is None:
#             hidden_dims = [256, 128, 64]

#         layers = []
#         prev = input_dim
#         for h in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev, h),
#                 nn.BatchNorm1d(h),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#             ])
#             prev = h

#         self.backbone = nn.Sequential(*layers)
#         self.head = nn.Linear(prev, num_classes)
#         self.embed_dim = prev          # exposed for fusion

#     def get_embedding(self, x):
#         return self.backbone(x)

#     def forward(self, x):
#         return self.head(self.get_embedding(x))


# # ================================================================
# # MODEL 2: RETINAL EFFICIENTNET-B3
# # ================================================================
# class RetinalEfficientNet(nn.Module):
#     """
#     PLACEHOLDER — Match your Colab EfficientNet-B3 head exactly.
    
#     To find your architecture, run this in Colab:
#         print(model)
#         print(model.classifier)  # or model._fc
#     """
#     def __init__(self, num_classes=5, pretrained=False):
#         super(RetinalEfficientNet, self).__init__()
#         from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

#         if pretrained:
#             self.base = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
#         else:
#             self.base = efficientnet_b3(weights=None)

#         # B3 final features = 1536
#         in_features = self.base.classifier[1].in_features
#         self.base.classifier = nn.Identity()

#         self.head = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_features, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes),
#         )
#         self.embed_dim = 256

#     def get_embedding(self, x):
#         features = self.base(x)                   # (B, 1536)
#         # Through dropout + first linear + ReLU (skip final linear)
#         x = self.head[0](features)                # Dropout
#         x = self.head[1](x)                       # Linear(1536→256)
#         x = self.head[2](x)                       # ReLU
#         return x                                   # (B, 256)

#     def forward(self, x):
#         features = self.base(x)
#         return self.head(features)


# # ================================================================
# # MODEL 3: GLOBAL FUSION
# # ================================================================
# class GlobalFusionModel(nn.Module):
#     """
#     PLACEHOLDER — Match your Colab fusion model exactly.
#     """
#     def __init__(self, ehr_embed_dim=64, retinal_embed_dim=256, num_classes=5):
#         super(GlobalFusionModel, self).__init__()
#         fused = ehr_embed_dim + retinal_embed_dim

#         self.fusion = nn.Sequential(
#             nn.Linear(fused, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes),
#         )

#     def forward(self, ehr_emb, retinal_emb):
#         return self.fusion(torch.cat([ehr_emb, retinal_emb], dim=1))