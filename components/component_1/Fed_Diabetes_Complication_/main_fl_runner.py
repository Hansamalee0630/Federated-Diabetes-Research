import torch
import pandas as pd
import numpy as np
import copy
import os
import json
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve

# Resolve paths relative to this file so artifacts stay inside the project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE_DIR, "experiments", "comp1_experiments")
RESULTS_MODELS_DIR = os.path.join(BASE_DIR, "results", "comp1_results", "models")

# Import local research components
from component.component_1.model_architectures import NephropathyNet, CVDNet
from fl_core.server import FederatedServer
from fl_core.client import DiabetesClient
from datasets.complication_dataset.data_preprocessing import run_preprocessing
from datasets.complication_dataset.intermediate_gen import generate_bridge_data

# NEW: REPRODUCIBILITY SEEDING
def set_seed(seed=50):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# NEW: ARTIFACT SAVING HELPERS
# ---------------------------------------------------------
def save_fl_results(history, filename):
    """Saves the training history to a JSON file."""
    os.makedirs(EXP_DIR, exist_ok=True)
    path = os.path.join(EXP_DIR, filename)
    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print(f" -> Results saved to {path}")

# ---------------------------------------------------------
# HELPER: Data Splitting with Fuzzy Weight Support
# ---------------------------------------------------------
def prepare_federated_data(csv_path, feature_cols, target_col, weight_col, num_hospitals=3):
    """
    Chunks the dataset into N hospital loaders.
    Includes weights for handling clinical confidence during training.
    """
    df = pd.read_csv(csv_path)
    
    X = torch.tensor(df[feature_cols].values).float()
    y = torch.tensor(df[target_col].values).float().view(-1, 1)
    w = torch.tensor(df[weight_col].values).float().view(-1, 1) 
    
    # Dataset returns 3 values: Features, Labels, and Fuzzy Weights
    dataset = TensorDataset(X, y, w)
    
    total_size = len(dataset)
    split_size = total_size // num_hospitals
    lengths = [split_size] * (num_hospitals - 1)
    lengths.append(total_size - sum(lengths))
    
    subsets = torch.utils.data.random_split(dataset, lengths)
    return {i: DataLoader(subsets[i], batch_size=16, shuffle=True) for i in range(num_hospitals)}

# ---------------------------------------------------------
# Evaluation helper - YOUDEN'S J OPTIMIZATION
# ---------------------------------------------------------
def evaluate_model(model, csv_path, feature_cols, label_col, device=None, batch_size=64):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)
    X = torch.tensor(df[feature_cols].values).float()
    y = torch.tensor(df[label_col].values).float().view(-1, 1)

    dl = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    model.to(device).eval()
    probs, trues = [], []
    
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb)).view(-1)
            probs.extend(out.cpu().numpy().tolist())
            trues.extend(yb.view(-1).cpu().numpy().tolist())

    # Calculate Optimal Threshold using Youden's J Statistic (TPR - FPR)
    # This specifically addresses the class imbalance you observed
    fpr, tpr, thresholds = roc_curve(trues, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    preds = [1 if p >= optimal_threshold else 0 for p in probs]
    
    auc = roc_auc_score(trues, probs) if len(set(trues)) > 1 else None
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average='binary', zero_division=0)
    cm = confusion_matrix(trues, preds)

    print(f"\n" + "="*60)
    print(f"RESEARCH EVALUATION: {os.path.basename(csv_path)}")
    print(f"="*60)
    print(f"Metrics: AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    #print(f"Threshold Optimization: Youden's J found Best Threshold at {optimal_threshold:.4f}")
    
    print(f"\n{' ':10} | Predicted Healthy | Predicted Risk")
    print(f"{'-'*58}")
    print(f"Actual Healthy (0)  | {cm[0][0]:^17} | {cm[0][1]:^14}")
    print(f"Actual Risk (1)     | {cm[1][0]:^17} | {cm[1][1]:^14}")
    print(f"{'-'*58}")
    
    return {"auc": auc, "acc": acc, "f1": f1, "threshold": optimal_threshold}

# ---------------------------------------------------------
# MASTER WORKFLOW
# ---------------------------------------------------------
def run_research_experiment():
    set_seed(50)
    print("="*50)
    print("STARTING OPTIMIZED FEDERATED LEARNING PIPELINE")
    print("="*50)

    os.makedirs(RESULTS_MODELS_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    # Step 1: Preprocessing with Fuzzy Weights
    print("\n[Step 1] Running Data Preprocessing...")
    run_preprocessing()
    
    processed_csv = os.path.join(BASE_DIR, "datasets", "complication_dataset", "processed_data.csv")
    nephro_features = ['AGE', 'BMI', 'HBA1C', 'CR', 'UREA'] 

    # Step 2: Stage 1 Training (Nephropathy)
    print(f"\n[Step 2] Training Stage 1: Nephropathy Model (15 Rounds)...")
    
    stage1_loaders = prepare_federated_data(processed_csv, nephro_features, "LABEL_NEPHROPATHY", "NEPH_WEIGHT")
    
    global_model_n = NephropathyNet(input_size=len(nephro_features))
    server_n = FederatedServer(global_model_n)
    
    # Clients are initialized with PrivacyEngine and Weighted Loss support
    clients_n = [DiabetesClient(i, copy.deepcopy(global_model_n), stage1_loaders[i]) for i in range(3)]

    nephro_history = []
    for r in range(15): 
        # local_train now returns (weights, avg_loss)
        results = [c.local_train(server_n.global_model.state_dict()) for c in clients_n]
        local_updates = [res[0] for res in results]
        avg_loss = np.mean([res[1] for res in results])
        
        server_n.aggregate_weights(local_updates)
        print(f"Round {r+1:02d}/15 | Aggregated Loss: {avg_loss:.4f}")

    evaluate_model(server_n.global_model, processed_csv, nephro_features, "LABEL_NEPHROPATHY")
    
    
    save_fl_results(nephro_history, "nephropathy_training_log.json")
    torch.save(server_n.global_model.state_dict(), os.path.join(EXP_DIR, "final_nephropathy_model.pth"))

    # Step 3: Sequential Dependency Bridge
    print("\n[Step 3] Generating Risk Scores (Nephropathy -> CVD)...")
    generate_bridge_data()
    bridge_csv = os.path.join(BASE_DIR, "datasets", "complication_dataset", "patient_data_with_nephro_score.csv")

    # Step 4: Stage 2 Training (CVD)
    cvd_features = ['CHOL', 'TG', 'HDL', "NEPHRO_RISK_SCORE"]
    print(f"\n[Step 4] Training Stage 2: CVD Risk Model (17 Rounds)...")
    stage2_loaders = prepare_federated_data(bridge_csv, cvd_features, "LABEL_CVD", "CVD_WEIGHT")
    
    global_model_c = CVDNet(input_size=len(cvd_features))
    server_c = FederatedServer(global_model_c)
    clients_c = [DiabetesClient(i, copy.deepcopy(global_model_c), stage2_loaders[i]) for i in range(3)]
    
    cvd_history = []
    for r in range(17): 
        results = [c.local_train(server_c.global_model.state_dict()) for c in clients_c]
        local_updates = [res[0] for res in results]
        avg_loss = np.mean([res[1] for res in results])
        
        server_c.aggregate_weights(local_updates)
        print(f"Round {r+1:02d}/17 | Aggregated Loss: {avg_loss:.4f}")

    evaluate_model(server_c.global_model, bridge_csv, cvd_features, "LABEL_CVD")

    save_fl_results(cvd_history, "cvd_training_log.json")
    torch.save(server_c.global_model.state_dict(), os.path.join(EXP_DIR, "final_cvd_model.pth"))

    
    print("\nPIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    run_research_experiment()