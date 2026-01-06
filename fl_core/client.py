# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import copy
# import os
# from sklearn.metrics import (
#     roc_auc_score, average_precision_score, f1_score, 
#     recall_score, precision_score, confusion_matrix, accuracy_score
# )
# import warnings
# from sklearn.exceptions import UndefinedMetricWarning
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# class FederatedClient:
#     def __init__(self, client_id, component_type="generic"):
#         self.client_id = client_id
#         self.component = component_type
#         self.data_path = f"datasets/diabetes_130/processed/client_{client_id}"
#         self.model = None
#         self.train_loader = None
        
#     def load_data(self, data_client_id=None):
#         """
#         Loads data dynamically based on the component type.
#         :param data_client_id: If provided, loads data from this ID instead of self.client_id.
#                                Useful for scalability testing (reusing data).
#         """
#         # If data_client_id is passed (e.g. 0), we load client_0 data even if self.client_id is 5.
#         file_id = data_client_id if data_client_id is not None else self.client_id
        
#         target_data_path = f"datasets/diabetes_130/processed/client_{file_id}"
#         self.data_path = target_data_path

#         print(f"Client {self.client_id}: Loading data for {self.component} (Source: Client {file_id})...")
#         dataset = None


#         if self.component in ["comp4_multitask", "comp4_singletask_htn", "comp4_singletask_hf"]:
#             try:
#                 if not os.path.exists(f"{target_data_path}_X.csv"):
#                     print(f"❌ Error: Data file not found: {target_data_path}_X.csv")
#                     print("   -> Run 'python datasets/diabetes_130/preprocess.py'?")

#                     self.train_loader = []
#                     return

#                 X = pd.read_csv(f"{target_data_path}_X.csv").values
#                 y = pd.read_csv(f"{target_data_path}_y.csv")
                
#                 # CASE A: Single Task - Hypertension
#                 if self.component == "comp4_singletask_htn":
#                     y_target = y['target_hypertension'].values

#                 # CASE B: Single Task - Heart Failure
#                 elif self.component == "comp4_singletask_hf":
#                     y_target = y['target_heart_failure'].values

#                 # CASE C: Multi-Task
#                 else:
#                     y_target = y[['target_hypertension', 'target_heart_failure', 'target_cluster']].values

#                 # Convert to Tensors for DataLoader
#                 tensor_x = torch.Tensor(X)
#                 tensor_y = torch.Tensor(y_target)
                
#                 dataset = TensorDataset(tensor_x, tensor_y)
                
#             except Exception as e:
#                 print(f"❌ Error loading Diabetes 130 data: {e}")
#                 self.train_loader = []
#                 return
            
#         else:
#             print(f"❌ Unknown component type: {self.component}")
#             return

#         # Finalize the Loader
#         if dataset:
#             self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#             print(f"   -> Data loaded successfully. ({len(dataset)} samples)")
#         else:
#             self.train_loader = [] # Safety fallback

    
#     # Receives the global model from the server.
#     def set_model(self, model):
#         self.model = model

#     def train(self, epochs=1):
#         """Local training loop."""
#         if not self.model:
#             raise ValueError("Model not set!")
        

#         if not self.train_loader or len(self.train_loader) == 0:
#             return self.model.state_dict() # Return unchanged weights

#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
#         # Loss Functions
#         criterion_binary = torch.nn.BCELoss()           # For Disease (0/1)
#         criterion_multi  = torch.nn.CrossEntropyLoss()  # For Cluster (0,1,2)
        
#         self.model.train()

#         for epoch in range(epochs):
#             for batch_X, batch_y in self.train_loader:
#                 optimizer.zero_grad()
                
#                 # --- MULTI-TASK LEARNING ---
#                 if self.component == "comp4_multitask":
#                     # The model returns 3 outputs
#                     pred_htn, pred_hf, pred_cluster = self.model(batch_X)
                    
#                     # The target batch also has 3 columns
#                     target_htn = batch_y[:, 0].unsqueeze(1)
#                     target_hf = batch_y[:, 1].unsqueeze(1)
#                     target_cluster = batch_y[:, 2].long()

#                     # 3. Calculate loss for all three tasks and add them together
#                     loss_htn = criterion_binary(pred_htn, target_htn)
#                     loss_hf = criterion_binary(pred_hf, target_hf)
#                     loss_cluster = criterion_multi(pred_cluster, target_cluster)
                    
#                     # 4. Total Loss (Sum them up)
#                     loss = loss_htn + loss_hf + loss_cluster

#                 # --- GENERIC COMPONENT (Single-Task) ---
#                 else:
#                     outputs = self.model(batch_X)
#                     loss = criterion_binary(outputs, batch_y.view(-1, 1))

#                 loss.backward()
#                 optimizer.step()
        
#         return self.model.state_dict() # Return weights to server

#     def train_with_privacy(self, epochs=1, epsilon=1.0, max_grad_norm=1.0):
#         """
#         Local training loop. Simulates Differential Privacy by clipping gradients and adding noise.
#         This fulfills the "Privacy-Preserving" requirement of the architecture.
#         Returns the updated model weights after training.
#         """
#         if not self.model:
#             raise ValueError("Model not set!")
        

#         if not self.train_loader or len(self.train_loader) == 0:
#             return self.model.state_dict() # Return unchanged weights

#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
#         # Loss Functions
#         criterion_binary = torch.nn.BCELoss()           # For Disease (0/1)
#         criterion_multi  = torch.nn.CrossEntropyLoss()  # For Cluster (0,1,2)
        
#         self.model.train()

#         # --- 1. TRAINING LOOP (With Gradient Clipping) ---
#         for epoch in range(epochs):
#             for batch_X, batch_y in self.train_loader:
#                 optimizer.zero_grad()
                
#                 # --- MULTI-TASK LEARNING ---
#                 if self.component == "comp4_multitask":
#                     pred_htn, pred_hf, pred_cluster = self.model(batch_X)
                    
#                     target_htn = batch_y[:, 0].unsqueeze(1)
#                     target_hf = batch_y[:, 1].unsqueeze(1)
#                     target_cluster = batch_y[:, 2].long()

#                     loss_htn = criterion_binary(pred_htn, target_htn)
#                     loss_hf = criterion_binary(pred_hf, target_hf)
#                     loss_cluster = criterion_multi(pred_cluster, target_cluster)
                    
#                     loss = loss_htn + loss_hf + loss_cluster

#                 # --- GENERIC COMPONENT (Single-Task) ---
#                 else:
#                     outputs = self.model(batch_X)
#                     loss = criterion_binary(outputs, batch_y.view(-1, 1))

#                 loss.backward()

#                 # --- PRIVACY STEP 1: CLIP GRADIENTS ---
#                 # This limits the influence of any single data point for every batch
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

#                 optimizer.step()

#         # --- PRIVACY STEP 2: ADD NOISE TO WEIGHTS (Local DP Simulation) ---
#         # Get the state dict (weights)
#         model_state = self.model.state_dict()
#         noise_multiplier = 1.0 / epsilon # Simplistic mapping for simulation

#         # Clone to avoid modifying the model for the next round of local training
#         noisy_state = {k: v.clone() for k, v in model_state.items()}

#         for key in noisy_state.keys():
#             if 'float' in str(noisy_state[key].dtype): # Only add noise to float parameters
#                 noise = torch.normal(0, 0.01 * noise_multiplier, size=noisy_state[key].shape)
#                 noisy_state[key] += noise

#         return noisy_state # Return the NOISY weights to server

#     def _evaluate_metrics(self, model_to_test):
#         """
#         Calculates AUROC, AUPRC, F1, Sensitivity, Specificity per task.
#         Returns a Dictionary of metrics.
#         """
#         if not self.train_loader: return {}

#         model_to_test.eval()
        
#         # Containers to store ALL predictions and targets for the whole dataset
#         all_preds_htn = []
#         all_targets_htn = []
#         all_preds_hf = []
#         all_targets_hf = []
#         all_preds_cluster = [] # Logits
#         all_targets_cluster = []

#         with torch.no_grad():
#             for batch_X, batch_y in self.train_loader:
#                 if self.component == "comp4_multitask":
#                     p_htn, p_hf, p_cluster = model_to_test(batch_X)
                    
#                     # Store HTN (Binary)
#                     all_preds_htn.extend(p_htn.numpy().flatten())
#                     all_targets_htn.extend(batch_y[:, 0].numpy().flatten())
                    
#                     # Store HF (Binary)
#                     all_preds_hf.extend(p_hf.numpy().flatten())
#                     all_targets_hf.extend(batch_y[:, 1].numpy().flatten())
                    
#                     # Store Cluster (Multi-class Logits)
#                     all_preds_cluster.extend(p_cluster.numpy())
#                     all_targets_cluster.extend(batch_y[:, 2].numpy())
                
#                 # --- CASE 2: SINGLE-TASK (HTN) ---
#                 elif self.component == "comp4_singletask_htn":
#                     # Model outputs only one value (HTN prob)
#                     pred = model_to_test(batch_X)
#                     all_preds_htn.extend(pred.numpy().flatten())
#                     # In Single Task HTN, y is just the column, so we take it directly
#                     all_targets_htn.extend(batch_y.numpy().flatten())

#                 # --- CASE 3: SINGLE-TASK (HF) ---
#                 else:
#                     pred = model_to_test(batch_X)
#                     all_preds_hf.extend(pred.numpy().flatten())
#                     all_targets_hf.extend(batch_y.numpy().flatten())

#         # --- Helper for Binary Metrics ---
#         def calc_binary_metrics(y_true, y_prob, prefix):
#             if len(y_true) == 0: return {}

#             y_true = np.array(y_true)
#             y_prob = np.array(y_prob)
#             y_pred = (y_prob > 0.5).astype(int)
            
#             # Handling rare cases where only one class exists in the batch
#             try:
#                 auroc = roc_auc_score(y_true, y_prob)
#             except: auroc = 0.5 
            
#             try:
#                 auprc = average_precision_score(y_true, y_prob)
#             except: auprc = 0.0

#             # Confusion Matrix handling for safety
#             try:
#                 tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
#                 sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#                 specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#             except:
#                 sensitivity = 0
#                 specificity = 0
            
#             return {
#                 f"{prefix}_acc": accuracy_score(y_true, y_pred),
#                 f"{prefix}_auroc": auroc,
#                 f"{prefix}_auprc": auprc,
#                 f"{prefix}_f1": f1_score(y_true, y_pred),
#                 f"{prefix}_sens": sensitivity,
#                 f"{prefix}_spec": specificity
#             }

#         metrics = {}
        
#         # 1. Hypertension Metrics - Calculate Hypertension Metrics (Only if we have data for it)
#         if len(all_targets_htn) > 0:
#             metrics.update(calc_binary_metrics(all_targets_htn, all_preds_htn, "htn"))
#             metrics['overall_acc'] = metrics['htn_acc'] # Default for single task

#         # 2. Heart Failure Metrics - Calculate Heart Failure Metrics
#         if len(all_targets_hf) > 0:
#             metrics.update(calc_binary_metrics(all_targets_hf, all_preds_hf, "hf"))
#             metrics['overall_acc'] = metrics['hf_acc'] # Override if this is the task

#         # 3. Cluster Metrics (Multi-class)- Convert Logits to Probabilities -> Then to Labels
#         # Calculate Cluster Metrics (ONLY FOR MULTI-TASK)
#         if len(all_targets_cluster) > 0:
#             cluster_logits = np.array(all_preds_cluster)
#             cluster_probs = torch.softmax(torch.tensor(cluster_logits), dim=1).numpy()
#             cluster_preds = np.argmax(cluster_probs, axis=1)
#             cluster_targets = np.array(all_targets_cluster)
            
#             metrics['cluster_acc'] = accuracy_score(cluster_targets, cluster_preds)
#             metrics['cluster_f1_macro'] = f1_score(cluster_targets, cluster_preds, average='macro')
            
#             # Recalculate Overall Acc for Multi-task (Avg of 3)
#             if 'htn_acc' in metrics and 'hf_acc' in metrics:
#                 metrics['overall_acc'] = (metrics['htn_acc'] + metrics['hf_acc'] + metrics['cluster_acc']) / 3.0
        
#         return metrics

#     def evaluate_personalization(self, epochs=10):
#         """
#         Research Objective 2.1 & 2.2:
#         Fine-tunes the global model on local data to measure Personalization Gain.
#         Returns: Accuracy BEFORE fine-tuning vs. Accuracy AFTER fine-tuning.
#         Research Objective 2.1: Adaptive Parameter Sharing.
#         Freezes the 'Shared Body' and only fine-tunes the 'Task Heads'.
#         """
#         if not self.model or not self.train_loader or len(self.train_loader) == 0:
#             return 0.0, 0.0

#         # 1. Measure Baseline (Global Model Performance)
#         baseline_metrics = self._evaluate_metrics(self.model)

#         # 2. Personalization Step (Fine-Tuning)
#         personalized_model = copy.deepcopy(self.model)
        
#         # === THE "ADAPTIVE" LOGIC ===
#         # We iterate through layers. If it's a "shared" layer, we freeze it.
#         # If it's a "head" layer, we let it learn.
#         for name, param in personalized_model.named_parameters():
#             if "shared" in name or "bn" in name: # Freeze Body & Batch Norm
#                 param.requires_grad = False
#             else: # Train Heads
#                 param.requires_grad = True
        
#         # Only optimize parameters that require gradient
#         # Add L2 Regularization (Weight Decay) to prevent overfitting during personalization
#         optimizer = torch.optim.Adam(
#             filter(lambda p: p.requires_grad, personalized_model.parameters()), 
#             lr=0.001, 
#             weight_decay = 1e-5  # ADDS L2 REGULARIZATION
#         )
#         criterion_binary = torch.nn.BCELoss()
#         criterion_multi  = torch.nn.CrossEntropyLoss()

#         # 3. Training Loop (Standard)
#         personalized_model.train()
#         for epoch in range(epochs):
#             for batch_X, batch_y in self.train_loader:
#                 optimizer.zero_grad()
                
#                 if self.component == "comp4_multitask":
#                     pred_htn, pred_hf, pred_cluster = personalized_model(batch_X)
                    
#                     loss = criterion_binary(pred_htn, batch_y[:, 0].unsqueeze(1)) + \
#                            criterion_binary(pred_hf, batch_y[:, 1].unsqueeze(1)) + \
#                            criterion_multi(pred_cluster, batch_y[:, 2].long())
#                 else:
#                     pred = personalized_model(batch_X)
#                     loss = criterion_binary(pred, batch_y.view(-1, 1))
                    
#                 loss.backward()
#                 optimizer.step()
                
#         # 4. Measure Personalized Performance. Swap models temporarily
#         temp_model = self.model 
#         self.model = personalized_model 
#         personalized_metrics = self._evaluate_metrics(personalized_model)
#         self.model = temp_model # Swap back
        
#         return baseline_metrics, personalized_metrics

#     def _calculate_accuracy(self):
#         """Helper to calculate accuracy on local data."""
#         if not self.train_loader or len(self.train_loader) == 0:
#             return 0.0

#         correct = 0
#         total = 0
#         self.model.eval()
#         with torch.no_grad():
#             for batch_X, batch_y in self.train_loader:
#                 if self.component == "comp4_multitask":
#                     p_htn, p_hf, p_cluster = self.model(batch_X)

#                     # Average accuracy of both tasks (Binary Accuracy)
#                     acc_htn = ((p_htn > 0.5) == (batch_y[:, 0].unsqueeze(1) > 0.5)).sum().item()
#                     acc_hf  = ((p_hf > 0.5) == (batch_y[:, 1].unsqueeze(1) > 0.5)).sum().item()
                    
#                     # Multi-Class Accuracy (Cluster)
#                     # p_cluster shape: [Batch, 3] -> Get index of max value
#                     _, pred_cluster_labels = torch.max(p_cluster, 1)
#                     acc_cluster = (pred_cluster_labels == batch_y[:, 2].long()).sum().item()
                    
#                     # Average accuracy of all 3 tasks
#                     correct += (acc_htn + acc_hf + acc_cluster) / 3 # Avg correct
#                 else:
#                     outputs = self.model(batch_X)
#                     predicted = (outputs > 0.5).float()
#                     correct += (predicted == batch_y.view(-1, 1)).sum().item()
#                 total += batch_y.size(0)
        
#         if total == 0: return 0.0
#         return correct / total
    
#     def evaluate_fairness(self):
#         """
#         Research Objective 2.2.iii & Target 4:
#         Calculate Demographic Parity Gap (Fairness).
#         Handles Standardized Data (where values are not exactly 0 or 1).
#         """
#         if not self.model: return 0.0
        
#         try:
#              # Read the file currently assigned to this client.
#              if not os.path.exists(f"{self.data_path}_X.csv"):
#                  return 0.0
#              else:
#                 df = pd.read_csv(f"{self.data_path}_X.csv")
#                 y = pd.read_csv(f"{self.data_path}_y.csv")
#         except:
#             return 0.0
        
#         # --- SMART COLUMN DETECTION (SCALING PROOF) ---
#         group_a = None # Females
#         group_b = None # Males

#         # We look for the column. Since data is scaled, we use > 0 or < 0
#         if 'gender_Female' in df.columns:
#             # If Scaled: > 0 is likely 1 (Female), < 0 is likely 0 (Male)
#             group_a = df[df['gender_Female'] > 0] # Females
#             group_b = df[df['gender_Female'] < 0] # Males
#         elif 'gender_Male' in df.columns:
#             # If Scaled: < 0 is likely 0 (Female), > 0 is likely 1 (Male)
#             group_a = df[df['gender_Male'] < 0] # Females
#             group_b = df[df['gender_Male'] > 0] # Males
#         else:
#             return 0.0

#         # Helper to get accuracy
#         def get_acc(subset_X, subset_indices):
#             if len(subset_X) == 0: return 0.0
#             subset_y = y.iloc[subset_indices]

#             X_vals = subset_X.astype(np.float32).values
#             t_X = torch.Tensor(X_vals)
            
#             self.model.eval()
#             with torch.no_grad():
#                 if self.component == "comp4_multitask":
#                     p_htn, _, _ = self.model(t_X) # Check ONLY Hypertension for fairness
#                     preds = (p_htn > 0.5).float().numpy()
#                     targets = subset_y['target_hypertension'].values.reshape(-1, 1)
#                 else:
#                     out = self.model(t_X)
#                     preds = (out > 0.5).float().numpy()
#                     targets = subset_y.iloc[:, 0].values.reshape(-1, 1)
            
#             return (preds == targets).mean()

#         # Calculate Accuracy
#         acc_female = get_acc(group_a, group_a.index)
#         acc_male = get_acc(group_b, group_b.index)
        
#         gap = abs(acc_female - acc_male)
        
#         return gap



import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    recall_score, precision_score, confusion_matrix, accuracy_score
)
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
DECISION_THRESHOLD = 0.5

class FederatedClient:
    def __init__(self, client_id, component_type="generic"):
        self.client_id = client_id
        self.component = component_type
        self.data_path = f"datasets/diabetes_130/processed/client_{client_id}"
        self.model = None
        self.train_loader = None
        # Initialize default weights
        self.htn_weight = 1.0
        self.hf_weight = 1.0
        
    def load_data(self, data_client_id=None):
        """
        Loads data dynamically based on the component type.
        Calculates CLASS WEIGHTS to handle imbalance.
        """
        file_id = data_client_id if data_client_id is not None else self.client_id
        
        target_data_path = f"datasets/diabetes_130/processed/client_{file_id}"
        self.data_path = target_data_path

        print(f"Client {self.client_id}: Loading data for {self.component} (Source: Client {file_id})...")
        dataset = None

        if self.component in ["comp4_multitask", "comp4_singletask_htn", "comp4_singletask_hf"]:
            try:
                if not os.path.exists(f"{target_data_path}_X.csv"):
                    print(f"Error: Data file not found: {target_data_path}_X.csv")
                    self.train_loader = []
                    return

                X = pd.read_csv(f"{target_data_path}_X.csv").values
                y = pd.read_csv(f"{target_data_path}_y.csv")
                
                y_target = None

                # CASE A: Single Task - Hypertension
                if self.component == "comp4_singletask_htn":
                    y_target = y['target_hypertension'].values

                # CASE B: Single Task - Heart Failure
                elif self.component == "comp4_singletask_hf":
                    y_target = y['target_heart_failure'].values

                # CASE C: Multi-Task
                else:
                    y_target = y[['target_hypertension', 'target_heart_failure', 'target_cluster']].values

                # --- DYNAMIC CLASS BALANCING (The Fix) ---
                num_samples = len(y_target)
                
                # 1. Hypertension Weight Calculation
                if self.component in ["comp4_multitask", "comp4_singletask_htn"]:
                    if self.component == "comp4_multitask":
                        htn_col = y_target[:, 0]
                    else:
                        htn_col = y_target # Single task is 1D
                        
                    num_pos_htn = np.sum(htn_col)
                    num_neg_htn = num_samples - num_pos_htn
                    # Weight = Negatives / Positives (Capped at 8.0 to prevent explosion)
                    self.htn_weight = min(float(num_neg_htn / (num_pos_htn + 1e-6)), 8.0)
                    print(f"     HTN Imbalance: {num_pos_htn} pos / {num_neg_htn} neg -> Weight: {self.htn_weight:.2f}")
                
                # 2. Heart Failure Weight Calculation
                if self.component in ["comp4_multitask", "comp4_singletask_hf"]:
                    if self.component == "comp4_multitask":
                        hf_col = y_target[:, 1]
                    else:
                        hf_col = y_target

                    num_pos_hf = np.sum(hf_col)
                    num_neg_hf = num_samples - num_pos_hf
                    self.hf_weight = min(float(num_neg_hf / (num_pos_hf + 1e-6)), 8.0)
                    print(f"     HF Imbalance:  {num_pos_hf} pos / {num_neg_hf} neg -> Weight: {self.hf_weight:.2f}")

                # Convert to Tensors
                tensor_x = torch.Tensor(X)
                tensor_y = torch.Tensor(y_target)
                
                dataset = TensorDataset(tensor_x, tensor_y)
                
            except Exception as e:
                print(f"Error loading Diabetes 130 data: {e}")
                self.train_loader = []
                return
            
        else:
            print(f"Unknown component type: {self.component}")
            return

        # Finalize the Loader
        if dataset:
            self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            print(f"   -> Data loaded successfully. ({len(dataset)} samples)")
        else:
            self.train_loader = []

    def set_model(self, model):
        self.model = model

    def train(self, epochs=20):
        """Local training loop with WEIGHTED LOSS."""
        if not self.model: raise ValueError("Model not set!")
        if not self.train_loader or len(self.train_loader) == 0:
            return self.model.state_dict()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion_multi = torch.nn.CrossEntropyLoss()
        
        # --- CUSTOM WEIGHTED LOSS FUNCTION ---
        def weighted_bce_loss(pred, target, weight):
            # Penalize missing a "1" (Sick) much more than missing a "0" (Healthy)
            # Formula: - [weight * y * log(p) + (1-y) * log(1-p)]
            loss = - (weight * target * torch.log(pred + 1e-7) + 
                     (1 - target) * torch.log(1 - pred + 1e-7))
            return torch.mean(loss)

        self.model.train()

        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                # --- MULTI-TASK LEARNING ---
                if self.component == "comp4_multitask":
                    pred_htn, pred_hf, pred_cluster = self.model(batch_X)
                    
                    target_htn = batch_y[:, 0].unsqueeze(1)
                    target_hf = batch_y[:, 1].unsqueeze(1)
                    target_cluster = batch_y[:, 2].long()

                    # Apply Dynamic Weights
                    loss_htn = weighted_bce_loss(pred_htn, target_htn, self.htn_weight)
                    loss_hf = weighted_bce_loss(pred_hf, target_hf, self.hf_weight)
                    loss_cluster = criterion_multi(pred_cluster, target_cluster)
                    
                    loss = loss_htn + loss_hf + loss_cluster

                # --- SINGLE-TASK ---
                else:
                    outputs = self.model(batch_X)
                    # Use the correct weight based on component name
                    w = self.htn_weight if "htn" in self.component else self.hf_weight
                    loss = weighted_bce_loss(outputs, batch_y.view(-1, 1), w)

                loss.backward()
                optimizer.step()
        
        return self.model.state_dict()

    def train_with_privacy(self, epochs=1, epsilon=1.0, max_grad_norm=1.0):
        """Local training loop with WEIGHTED LOSS + PRIVACY."""
        if not self.model: raise ValueError("Model not set!")
        if not self.train_loader or len(self.train_loader) == 0:
            return self.model.state_dict()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion_multi = torch.nn.CrossEntropyLoss()

        # Re-define weighted loss here
        def weighted_bce_loss(pred, target, weight):
            loss = - (weight * target * torch.log(pred + 1e-7) + 
                     (1 - target) * torch.log(1 - pred + 1e-7))
            return torch.mean(loss)
        
        self.model.train()

        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                if self.component == "comp4_multitask":
                    pred_htn, pred_hf, pred_cluster = self.model(batch_X)
                    target_htn = batch_y[:, 0].unsqueeze(1)
                    target_hf = batch_y[:, 1].unsqueeze(1)
                    target_cluster = batch_y[:, 2].long()

                    loss_htn = weighted_bce_loss(pred_htn, target_htn, self.htn_weight)
                    loss_hf = weighted_bce_loss(pred_hf, target_hf, self.hf_weight)
                    loss_cluster = criterion_multi(pred_cluster, target_cluster)
                    
                    loss = loss_htn + loss_hf + loss_cluster
                else:
                    outputs = self.model(batch_X)
                    w = self.htn_weight if "htn" in self.component else self.hf_weight
                    loss = weighted_bce_loss(outputs, batch_y.view(-1, 1), w)

                loss.backward()

                # Privacy: Clip Gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

        # Privacy: Add Noise
        model_state = self.model.state_dict()
        noise_multiplier = 1.0 / epsilon 
        noisy_state = {k: v.clone() for k, v in model_state.items()}

        for key in noisy_state.keys():
            if 'float' in str(noisy_state[key].dtype): 
                noise = torch.normal(0, 0.01 * noise_multiplier, size=noisy_state[key].shape)
                noisy_state[key] += noise

        return noisy_state

    def _evaluate_metrics(self, model_to_test):
        """Calculates metrics."""
        if not self.train_loader: return {}

        model_to_test.eval()
        
        all_preds_htn = []
        all_targets_htn = []
        all_preds_hf = []
        all_targets_hf = []
        all_preds_cluster = [] 
        all_targets_cluster = []

        with torch.no_grad():
            for batch_X, batch_y in self.train_loader:
                if self.component == "comp4_multitask":
                    p_htn, p_hf, p_cluster = model_to_test(batch_X)
                    all_preds_htn.extend(p_htn.numpy().flatten())
                    all_targets_htn.extend(batch_y[:, 0].numpy().flatten())
                    all_preds_hf.extend(p_hf.numpy().flatten())
                    all_targets_hf.extend(batch_y[:, 1].numpy().flatten())
                    all_preds_cluster.extend(p_cluster.numpy())
                    all_targets_cluster.extend(batch_y[:, 2].numpy())
                elif self.component == "comp4_singletask_htn":
                    pred = model_to_test(batch_X)
                    all_preds_htn.extend(pred.numpy().flatten())
                    all_targets_htn.extend(batch_y.numpy().flatten())
                else:
                    pred = model_to_test(batch_X)
                    all_preds_hf.extend(pred.numpy().flatten())
                    all_targets_hf.extend(batch_y.numpy().flatten())

        def calc_binary_metrics(y_true, y_prob, prefix):
            if len(y_true) == 0: return {}
            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            y_pred = (y_prob > DECISION_THRESHOLD).astype(int)
            
            try: auroc = roc_auc_score(y_true, y_prob)
            except: auroc = 0.5 
            try: auprc = average_precision_score(y_true, y_prob)
            except: auprc = 0.0
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            except: sensitivity = 0; specificity = 0
            
            # In _evaluate_metrics(), after computing predictions:

            print(f"[DEBUG] {prefix}: Pos Rate = {np.mean(y_true):.2f} | Pred Pos Rate = {np.mean(y_pred):.2f}")
            print(f"[DEBUG] {prefix}: F1={f1_score(y_true, y_pred):.3f} | Sensitivity={sensitivity:.3f}")

            return {
                f"{prefix}_acc": accuracy_score(y_true, y_pred),
                f"{prefix}_auroc": auroc,
                f"{prefix}_auprc": auprc,
                f"{prefix}_f1": f1_score(y_true, y_pred),
                f"{prefix}_sens": sensitivity,
                f"{prefix}_spec": specificity
            }

        metrics = {}

        if len(all_targets_htn) > 0:
            metrics.update(calc_binary_metrics(all_targets_htn, all_preds_htn, "htn"))
            metrics['overall_acc'] = metrics['htn_acc']
        if len(all_targets_hf) > 0:
            metrics.update(calc_binary_metrics(all_targets_hf, all_preds_hf, "hf"))
            metrics['overall_acc'] = metrics['hf_acc']
        if len(all_targets_cluster) > 0:
            cluster_logits = np.array(all_preds_cluster)
            cluster_probs = torch.softmax(torch.tensor(cluster_logits), dim=1).numpy()
            cluster_preds = np.argmax(cluster_probs, axis=1)
            cluster_targets = np.array(all_targets_cluster)
            metrics['cluster_acc'] = accuracy_score(cluster_targets, cluster_preds)
            metrics['cluster_f1_macro'] = f1_score(cluster_targets, cluster_preds, average='macro')
            if 'htn_acc' in metrics and 'hf_acc' in metrics:
                metrics['overall_acc'] = (metrics['htn_acc'] + metrics['hf_acc'] + metrics['cluster_acc']) / 3.0
        
        return metrics

    def evaluate_personalization(self, epochs=10):
        """Includes Weighted Loss for Fine-Tuning."""
        if not self.model or not self.train_loader or len(self.train_loader) == 0:
            return 0.0, 0.0

        baseline_metrics = self._evaluate_metrics(self.model)
        personalized_model = copy.deepcopy(self.model)
        
        for name, param in personalized_model.named_parameters():
            if "shared" in name or "bn" in name: 
                param.requires_grad = False
            else: 
                param.requires_grad = True
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, personalized_model.parameters()), 
            lr=0.001, weight_decay = 1e-5
        )
        criterion_multi = torch.nn.CrossEntropyLoss()
        
        # Redefine weighted loss for personalization scope
        def weighted_bce_loss(pred, target, weight):
            loss = - (weight * target * torch.log(pred + 1e-7) + 
                     (1 - target) * torch.log(1 - pred + 1e-7))
            return torch.mean(loss)

        personalized_model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                if self.component == "comp4_multitask":
                    pred_htn, pred_hf, pred_cluster = personalized_model(batch_X)
                    loss = weighted_bce_loss(pred_htn, batch_y[:, 0].unsqueeze(1), self.htn_weight) + \
                           weighted_bce_loss(pred_hf, batch_y[:, 1].unsqueeze(1), self.hf_weight) + \
                           criterion_multi(pred_cluster, batch_y[:, 2].long())
                else:
                    pred = personalized_model(batch_X)
                    w = self.htn_weight if "htn" in self.component else self.hf_weight
                    loss = weighted_bce_loss(pred, batch_y.view(-1, 1), w)
                    
                loss.backward()
                optimizer.step()
                
        temp_model = self.model 
        self.model = personalized_model 
        personalized_metrics = self._evaluate_metrics(personalized_model)
        self.model = temp_model 
        
        return baseline_metrics, personalized_metrics

    def _calculate_accuracy(self):
        """Helper to calculate accuracy on local data."""
        if not self.train_loader or len(self.train_loader) == 0: return 0.0
        correct = 0; total = 0
        self.model.eval()
        
        with torch.no_grad():
            
            for batch_X, batch_y in self.train_loader:
                
                if self.component == "comp4_multitask":
                    p_htn, p_hf, p_cluster = self.model(batch_X)
                    acc_htn = ((p_htn > 0.5) == (batch_y[:, 0].unsqueeze(1) > 0.5)).sum().item()
                    acc_hf  = ((p_hf > 0.5) == (batch_y[:, 1].unsqueeze(1) > 0.5)).sum().item()
                    _, pred_cluster_labels = torch.max(p_cluster, 1)
                    acc_cluster = (pred_cluster_labels == batch_y[:, 2].long()).sum().item()
                    correct += (acc_htn + acc_hf + acc_cluster) / 3 
                else:
                    outputs = self.model(batch_X)
                    predicted = (outputs > DECISION_THRESHOLD).float()
                    correct += (predicted == batch_y.view(-1, 1)).sum().item()
                total += batch_y.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_fairness(self):
        """Calculates Demographic Parity Gap."""
        if not self.model: return 0.0
        
        try:
             if not os.path.exists(f"{self.data_path}_X.csv"): return 0.0
             else:
                df = pd.read_csv(f"{self.data_path}_X.csv")
                y = pd.read_csv(f"{self.data_path}_y.csv")
        except: return 0.0
        
        group_a = None # Females
        group_b = None # Males

        # if 'gender_Female' in df.columns:
        #     group_a = df[df['gender_Female'] > 0]; group_b = df[df['gender_Female'] < 0]
        # elif 'gender_Male' in df.columns:
        #     group_a = df[df['gender_Male'] < 0]; group_b = df[df['gender_Male'] > 0]
        # else: return 0.0

        if 'gender_Female' in df.columns:
            # If One-Hot encoded, usually 1=Yes (Female), 0=No (Male)
            group_a = df[df['gender_Female'] == 1] 
            group_b = df[df['gender_Female'] == 0] 
        elif 'gender_Male' in df.columns:
            # 1=Male, 0=Female
            group_a = df[df['gender_Male'] == 0] # Female
            group_b = df[df['gender_Male'] == 1] # Male
        else:
            return 0.0 # Column not found

        def get_acc(subset_X, subset_indices):
            if len(subset_X) == 0: return 0.0
            subset_y = y.iloc[subset_indices]
            
            t_X = torch.Tensor(subset_X.astype(np.float32).values)
            self.model.eval()
            with torch.no_grad():
                if self.component == "comp4_multitask":
                    p_htn, _, _ = self.model(t_X)
                    preds = (p_htn > 0.5).float().numpy()
                    targets = subset_y['target_hypertension'].values.reshape(-1, 1)
                else:
                    out = self.model(t_X)
                    preds = (out > 0.5).float().numpy()
                    targets = subset_y.iloc[:, 0].values.reshape(-1, 1)
            
            return (preds == targets).mean()

        acc_female = get_acc(group_a, group_a.index)
        acc_male = get_acc(group_b, group_b.index)

        # Debug print to verify fix (Optional)
        print(f"Client {self.client_id} Fairness: Female Acc {acc_female:.2f} | Male Acc {acc_male:.2f}")

        return abs(acc_female - acc_male)