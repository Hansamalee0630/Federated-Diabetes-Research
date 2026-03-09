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

        self.htn_weight = 1.0
        self.hf_weight = 1.0
        
    def load_data(self, data_client_id=None):
        # Calculates CLASS WEIGHTS to handle imbalance.
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

                if self.component == "comp4_singletask_htn":
                    y_target = y['target_hypertension'].values
                elif self.component == "comp4_singletask_hf":
                    y_target = y['target_heart_failure'].values
                else:
                    y_target = y[['target_hypertension', 'target_heart_failure', 'target_cluster']].values

                # DYNAMIC CLASS BALANCING
                num_samples = len(y_target)
                
                # 1. Hypertension Weight Calculation
                if self.component in ["comp4_multitask", "comp4_singletask_htn"]:
                    if self.component == "comp4_multitask":
                        htn_col = y_target[:, 0]
                    else:
                        htn_col = y_target
                        
                    num_pos_htn = np.sum(htn_col)
                    num_neg_htn = num_samples - num_pos_htn
                    # Weight = Negatives / Positives (Capped at 10.0 to prevent explosion)
                    self.htn_weight = min(float(num_neg_htn / (num_pos_htn + 1e-6)), 4.0)
                    print(f"     HTN Imbalance: {num_pos_htn} pos / {num_neg_htn} neg -> Weight: {self.htn_weight:.2f}")
                
                # 2. Heart Failure Weight Calculation
                if self.component in ["comp4_multitask", "comp4_singletask_hf"]:
                    if self.component == "comp4_multitask":
                        hf_col = y_target[:, 1]
                    else:
                        hf_col = y_target

                    num_pos_hf = np.sum(hf_col)
                    num_neg_hf = num_samples - num_pos_hf
                    self.hf_weight = min(float(num_neg_hf / (num_pos_hf + 1e-6)), 4.0)
                    print(f"     HF Imbalance:  {num_pos_hf} pos / {num_neg_hf} neg -> Weight: {self.hf_weight:.2f}")

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

        if dataset:
            self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            print(f"   -> Data loaded successfully. ({len(dataset)} samples)")
        else:
            self.train_loader = []

    def set_model(self, model):
        self.model = model


    # Local training loop with BCEWithLogitsLoss for class imbalance.
    def train(self, epochs=25):
        if not self.model: raise ValueError("Model not set!")
        if not self.train_loader or len(self.train_loader) == 0:
            return self.model.state_dict()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion_multi = torch.nn.CrossEntropyLoss()
        
        # BCEWithLogitsLoss with pos_weight for class imbalance
        # pos_weight > 1 penalizes missing positive cases (sick patients) more heavily
        criterion_htn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.htn_weight]))
        criterion_hf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hf_weight]))

        self.model.train()

        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                # MULTI-TASK LEARNING
                if self.component == "comp4_multitask":
                    # Model outputs raw logits (no sigmoid)
                    logits_htn, logits_hf, logits_cluster = self.model(batch_X)
                    
                    target_htn = batch_y[:, 0].unsqueeze(1)
                    target_hf = batch_y[:, 1].unsqueeze(1)
                    target_cluster = batch_y[:, 2].long()

                    # BCEWithLogitsLoss handles sigmoid internally
                    loss_htn = criterion_htn(logits_htn, target_htn)
                    loss_hf = criterion_hf(logits_hf, target_hf)
                    loss_cluster = criterion_multi(logits_cluster, target_cluster)
                    
                    loss = loss_htn + loss_hf + loss_cluster

                # SINGLE-TASK
                else:
                    logits = self.model(batch_X)
                    criterion = criterion_htn if "htn" in self.component else criterion_hf
                    loss = criterion(logits, batch_y.view(-1, 1))

                loss.backward()
                optimizer.step()
        
        return self.model.state_dict()


    # Local training loop with WEIGHTED LOSS + PRIVACY.
    def train_with_privacy(self, epochs=1, epsilon=1.0, max_grad_norm=1.0):
        if not self.model: raise ValueError("Model not set!")
        if not self.train_loader or len(self.train_loader) == 0:
            return self.model.state_dict()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion_multi = torch.nn.CrossEntropyLoss()
        
        # BCEWithLogitsLoss with pos_weight for class imbalance
        criterion_htn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.htn_weight]))
        criterion_hf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hf_weight]))
        
        self.model.train()

        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                if self.component == "comp4_multitask":
                    logits_htn, logits_hf, logits_cluster = self.model(batch_X)
                    target_htn = batch_y[:, 0].unsqueeze(1)
                    target_hf = batch_y[:, 1].unsqueeze(1)
                    target_cluster = batch_y[:, 2].long()

                    loss_htn = criterion_htn(logits_htn, target_htn)
                    loss_hf = criterion_hf(logits_hf, target_hf)
                    loss_cluster = criterion_multi(logits_cluster, target_cluster)
                    
                    loss = loss_htn + loss_hf + loss_cluster
                else:
                    logits = self.model(batch_X)
                    criterion = criterion_htn if "htn" in self.component else criterion_hf
                    loss = criterion(logits, batch_y.view(-1, 1))

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
                    # Model outputs raw logits - apply sigmoid for probabilities
                    logits_htn, logits_hf, logits_cluster = model_to_test(batch_X)
                    p_htn = torch.sigmoid(logits_htn)
                    p_hf = torch.sigmoid(logits_hf)
                    
                    all_preds_htn.extend(p_htn.numpy().flatten())
                    all_targets_htn.extend(batch_y[:, 0].numpy().flatten())
                    all_preds_hf.extend(p_hf.numpy().flatten())
                    all_targets_hf.extend(batch_y[:, 1].numpy().flatten())
                    all_preds_cluster.extend(logits_cluster.numpy())
                    all_targets_cluster.extend(batch_y[:, 2].numpy())
                elif self.component == "comp4_singletask_htn":
                    logits = model_to_test(batch_X)
                    pred = torch.sigmoid(logits)
                    all_preds_htn.extend(pred.numpy().flatten())
                    all_targets_htn.extend(batch_y.numpy().flatten())
                else:
                    logits = model_to_test(batch_X)
                    pred = torch.sigmoid(logits)
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


    # Includes Weighted Loss for Fine-Tuning.
    def evaluate_personalization(self, epochs=10):
        if not self.model or not self.train_loader or len(self.train_loader) == 0:
            return 0.0, 0.0

        baseline_metrics = self._evaluate_metrics(self.model)
        personalized_model = copy.deepcopy(self.model)
        
        for name, param in personalized_model.named_parameters():
            if "shared" in name or "bn" in name: 
                param.requires_grad = False
            else: 
                param.requires_grad = True
        
        # L2 Regularization via a weight decay
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, personalized_model.parameters()), 
            lr=0.001, weight_decay = 1e-5
        )
        criterion_multi = torch.nn.CrossEntropyLoss()
        
        # BCEWithLogitsLoss with pos_weight for class imbalance
        criterion_htn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.htn_weight]))
        criterion_hf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hf_weight]))

        personalized_model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                if self.component == "comp4_multitask":
                    logits_htn, logits_hf, logits_cluster = personalized_model(batch_X)
                    loss = criterion_htn(logits_htn, batch_y[:, 0].unsqueeze(1)) + \
                           criterion_hf(logits_hf, batch_y[:, 1].unsqueeze(1)) + \
                           criterion_multi(logits_cluster, batch_y[:, 2].long())
                else:
                    logits = personalized_model(batch_X)
                    criterion = criterion_htn if "htn" in self.component else criterion_hf
                    loss = criterion(logits, batch_y.view(-1, 1))
                    
                loss.backward()
                optimizer.step()
                
        temp_model = self.model 
        self.model = personalized_model 
        personalized_metrics = self._evaluate_metrics(personalized_model)
        self.model = temp_model 
        
        return baseline_metrics, personalized_metrics

    def _calculate_accuracy(self):
        if not self.train_loader or len(self.train_loader) == 0: return 0.0
        correct = 0; total = 0
        self.model.eval()
        
        with torch.no_grad():
            
            for batch_X, batch_y in self.train_loader:
                
                if self.component == "comp4_multitask":
                    # Model outputs raw logits - apply sigmoid
                    logits_htn, logits_hf, logits_cluster = self.model(batch_X)
                    p_htn = torch.sigmoid(logits_htn)
                    p_hf = torch.sigmoid(logits_hf)
                    
                    acc_htn = ((p_htn > 0.5) == (batch_y[:, 0].unsqueeze(1) > 0.5)).sum().item()
                    acc_hf  = ((p_hf > 0.5) == (batch_y[:, 1].unsqueeze(1) > 0.5)).sum().item()
                    _, pred_cluster_labels = torch.max(logits_cluster, 1)
                    acc_cluster = (pred_cluster_labels == batch_y[:, 2].long()).sum().item()
                    correct += (acc_htn + acc_hf + acc_cluster) / 3 
                else:
                    logits = self.model(batch_X)
                    probs = torch.sigmoid(logits)
                    predicted = (probs > DECISION_THRESHOLD).float()
                    correct += (predicted == batch_y.view(-1, 1)).sum().item()
                total += batch_y.size(0)
        
        return correct / total if total > 0 else 0.0
    
    # Calculates Demographic Parity Gap.
    def evaluate_fairness(self):
        if not self.model: return 0.0
        
        try:
             if not os.path.exists(f"{self.data_path}_X.csv"): return 0.0
             else:
                df = pd.read_csv(f"{self.data_path}_X.csv")
                y = pd.read_csv(f"{self.data_path}_y.csv")
        except: return 0.0
        
        group_a = None # Females
        group_b = None # Males

        if 'gender_Female' in df.columns: # If One-Hot encoded
            group_a = df[df['gender_Female'] == 1] 
            group_b = df[df['gender_Female'] == 0] 
        elif 'gender_Male' in df.columns:
            group_a = df[df['gender_Male'] == 0]
            group_b = df[df['gender_Male'] == 1]
        else:
            return 0.0

        def get_acc(subset_X, subset_indices):
            if len(subset_X) == 0: return 0.0
            subset_y = y.iloc[subset_indices]
            
            t_X = torch.Tensor(subset_X.astype(np.float32).values)
            self.model.eval()
            with torch.no_grad():
                if self.component == "comp4_multitask":
                    logits_htn, _, _ = self.model(t_X)
                    p_htn = torch.sigmoid(logits_htn)
                    preds = (p_htn > 0.5).float().numpy()
                    targets = subset_y['target_hypertension'].values.reshape(-1, 1)
                else:
                    logits = self.model(t_X)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float().numpy()
                    targets = subset_y.iloc[:, 0].values.reshape(-1, 1)
            
            return (preds == targets).mean()

        acc_female = get_acc(group_a, group_a.index)
        acc_male = get_acc(group_b, group_b.index)

        print(f"Client {self.client_id} Fairness: Female Acc {acc_female:.2f} | Male Acc {acc_male:.2f}")

        return abs(acc_female - acc_male)