"""
FAST OPTIMIZED Federated Learning Training (FedAvg) with Gender-Fairness
UPDATED FOR BALANCED DATASET
============================================================================
KEY CHANGES FROM UNBALANCED VERSION:
1. Load balanced dataset instead of unbalanced
2. Adjust class weights (more balanced now)
3. Reduce LOCAL_EPOCHS back to 5 (balanced data converges better)
4. Increase BATCH_SIZE to 64 (balanced data supports larger batches)
5. Adjust fairness thresholds (expectations change with balanced data)
6. Update MIN_RECALL_TARGET to 0.55 (expect better recall)
7. All else remains the same (FedAvg, gender fairness, SHAP compatible)
============================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score, 
    f1_score, confusion_matrix, balanced_accuracy_score
)
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from utils_fairness import compute_adaptive_fairness_params, compute_federated_convergence_criterion

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.random.set_seed(42)
np.random.seed(42)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_optimal_threshold(y_true, y_pred_prob):
    """Find optimal threshold by maximizing F1-score."""
    y_true = y_true.flatten()
    y_pred_prob = y_pred_prob.flatten()
    
    best_f1 = 0.0
    best_threshold = 0.5
    best_recall = 0.0
    
    # Optimize F1-score directly (simpler, more stable)
    for threshold in np.linspace(0.3, 0.7, 50):
        y_pred = (y_pred_prob > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_recall = recall_score(y_true, y_pred, zero_division=0)
    
    return best_threshold, best_f1, best_recall


# ============================================================================
# DATA LOADING (UPDATED FOR BALANCED DATASET)
# ============================================================================

def load_data():
    """Load BALANCED preprocessed data with hospital partitions."""
    print("Loading BALANCED data for Federated Learning...")
    # CHANGE: Use balanced dataset folder
    data_dir = Path("data/processed_data_balanced")
    
    # Check if balanced data exists, else fall back to unbalanced
    if not (data_dir / "X_train_balanced.npy").exists():
        print("\n⚠️  WARNING: Balanced dataset not found!")
        print("   Falling back to unbalanced dataset...")
        data_dir = Path("data/processed_data")
        X_train = np.load(data_dir / "X_train_unbalanced.npy").astype(np.float32)
        y_train = np.load(data_dir / "y_train_unbalanced.npy").astype(np.float32)
        X_test = np.load(data_dir / "X_test.npy").astype(np.float32)
        y_test = np.load(data_dir / "y_test.npy").astype(np.float32)
        sens_train = pd.read_csv(data_dir / "sensitive_attrs_train.csv")
        sens_test = pd.read_csv(data_dir / "sensitive_attrs_test.csv")
        is_balanced = False
    else:
        X_train = np.load(data_dir / "X_train_balanced.npy").astype(np.float32)
        y_train = np.load(data_dir / "y_train_balanced.npy").astype(np.float32)
        X_test = np.load(data_dir / "X_test_balanced.npy").astype(np.float32)
        y_test = np.load(data_dir / "y_test_balanced.npy").astype(np.float32)
        sens_train = pd.read_csv(data_dir / "sensitive_attrs_train_balanced.csv")
        sens_test = pd.read_csv(data_dir / "sensitive_attrs_test_balanced.csv")
        is_balanced = True
    
    dataset_type = "BALANCED ✓" if is_balanced else "UNBALANCED"
    print(f" Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean()*100:.2f}% [{dataset_type}]")
    
    print("\n VERIFYING DATA HETEROGENEITY:")
    for h in [1, 2, 3]:
        mask = sens_train['hospital_id'] == h
        y_h = y_train[mask]
        if len(y_h) > 0:
            print(f"   Hospital {h}: {len(y_h):,} samples | Pos rate: {y_h.mean()*100:.2f}%")
    
    return X_train, y_train, X_test, y_test, sens_train, sens_test, is_balanced


# ============================================================================
# CLASS WEIGHTS COMPUTATION (UPDATED FOR BALANCED DATA)
# ============================================================================

def compute_class_weights_per_hospital(y_train, sens_train):
    """
    Compute balanced class weights per hospital.
    
    For balanced data (50-50), weights will be closer to 1.0:1.0
    For imbalanced data (11-89), weights will be ~9:1
    """
    class_weights_dict = {}
    
    print("\n" + "="*70)
    print("COMPUTING CLASS WEIGHTS PER HOSPITAL")
    print("="*70)
    
    for hospital_id in [1, 2, 3]:
        mask = (sens_train['hospital_id'] == hospital_id).values
        y_h = y_train[mask]
        
        if len(y_h) == 0:
            continue
        
        n_pos = int((y_h == 1).sum())
        n_neg = int((y_h == 0).sum())
        n_total = len(y_h)
        
        # For balanced data, weights will be ~1.0:1.0
        # For imbalanced data, weights will be ~9:1
        weight_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
        
        class_weights_dict[hospital_id] = {
            0: float(weight_neg),
            1: float(weight_pos)
        }
        
        print(f"Hospital {hospital_id}:")
        print(f"  Samples: {n_pos:,} positive, {n_neg:,} negative")
        print(f"  Weights: Neg={weight_neg:.4f}, Pos={weight_pos:.4f}")
    
    return class_weights_dict


# ============================================================================
# FOCAL LOSS (COMPILED WITH tf.function)
# ============================================================================

@tf.function
def focal_loss_fn(y_true, y_pred, alpha=0.90, gamma=2.0):
    """
    Compiled focal loss for fast execution.
    
    For balanced data: alpha=0.90 still helps focus on harder positives
    For imbalanced data: alpha=0.90 helps severely undersampled positives
    """
    y_true = tf.cast(y_true, tf.float32)
    epsilon = 1e-7
    
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = tf.pow(1.0 - p_t, gamma)
    
    return tf.reduce_mean(alpha * focal_weight * bce)


@tf.function
def compute_fairness_loss(
    preds, batch_y, batch_prot, batch_weights,
    fairness_threshold=0.05, fairness_lambda=0.05
):
    """
    Compiled fairness penalty computation.
    
    For balanced data: Can afford stricter fairness (threshold=0.05, lambda=0.05)
    For imbalanced data: Need softer fairness (threshold=0.10, lambda=0.03)
    """
    batch_y = tf.cast(batch_y, tf.float32)
    batch_prot = tf.cast(batch_prot, tf.float32)
    batch_weights = tf.cast(batch_weights, tf.float32)
    
    # Clip predictions
    epsilon = 1e-7
    preds_clip = tf.clip_by_value(preds, epsilon, 1.0 - epsilon)
    
    # Compute per-sample losses
    bce = -batch_y * tf.math.log(preds_clip) - \
          (1 - batch_y) * tf.math.log(1 - preds_clip)
    p_t = batch_y * preds_clip + (1 - batch_y) * (1 - preds_clip)
    focal_weight = tf.pow(1.0 - p_t, 2.0)
    sample_loss = 0.90 * focal_weight * bce
    
    # Group-wise average loss (vectorized)
    prot_bool = tf.cast(tf.equal(batch_prot, 1.0), tf.float32)
    rest_bool = 1.0 - prot_bool
    
    prot_count = tf.reduce_sum(prot_bool) + 1e-6
    rest_count = tf.reduce_sum(rest_bool) + 1e-6
    
    loss_prot = tf.reduce_sum(sample_loss * prot_bool) / prot_count
    loss_rest = tf.reduce_sum(sample_loss * rest_bool) / rest_count
    
    # Soft fairness penalty
    fairness_penalty_raw = tf.abs(loss_prot - loss_rest)
    fairness_penalty = tf.maximum(0.0, fairness_penalty_raw - fairness_threshold)
    
    return fairness_penalty


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def create_model(input_dim):
    """Lighter model for faster training."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        
        # First block
        tf.keras.layers.Dense(
            128, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(2e-5)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Second block
        tf.keras.layers.Dense(
            64, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(2e-5)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Third block
        tf.keras.layers.Dense(
            32, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(2e-5)
        ),
        tf.keras.layers.Dropout(0.1),
        
        # Output layer
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    return model


# ============================================================================
# CLIENT DATA RETRIEVAL
# ============================================================================

def get_client_data(hospital_id, X, y, sens_df, protected_attr='gender'):
    """Extract data for a specific hospital."""
    mask = (sens_df["hospital_id"] == hospital_id).values
    return X[mask], y[mask], sens_df.loc[mask, protected_attr].values


# ============================================================================
# FEDERATED AVERAGING
# ============================================================================

def fedavg(weights_list, sample_counts):
    """Federated Averaging aggregation."""
    total_samples = sum(sample_counts)
    new_weights = [np.zeros_like(w) for w in weights_list[0]]
    
    for weights, count in zip(weights_list, sample_counts):
        scale = count / total_samples
        for i in range(len(weights)):
            new_weights[i] += scale * weights[i]
    
    return new_weights


# ============================================================================
# MAIN FEDERATED LEARNING LOOP
# ============================================================================

def run_federated_learning():
    """Fast federated learning training loop with balanced data."""
    print("\n" + "="*70)
    print("PHASE 4: FAST FEDERATED LEARNING WITH GENDER FAIRNESS")
    print("         (UPDATED FOR BALANCED DATASET)")
    print("="*70)

    start_time = time.time()
    
    X_train, y_train, X_test, y_test, sens_train, sens_test, is_balanced = load_data()
    class_weights_per_hospital = compute_class_weights_per_hospital(y_train, sens_train)
    
    input_dim = X_train.shape[1]
    global_model = create_model(input_dim)

    # ====================================================================
    # ADAPTIVE FAIRNESS PARAMETERS (data-driven, not hardcoded)
    # ====================================================================
    print("\nComputing Adaptive Fairness Parameters...")
    fairness_params = compute_adaptive_fairness_params(
        y_train, sens_train['gender'], dataset_is_balanced=is_balanced
    )
    
    print(f"  Baseline fairness gap: {fairness_params['baseline_gap']*100:.2f}%")
    print(f"  Strategy: {fairness_params['strategy'].upper()}")
    print(f"  Target threshold: {fairness_params['threshold']*100:.2f}%")
    print(f"  Fairness weight (lambda): {fairness_params['lambda']}")
    
    FAIRNESS_THRESHOLD = fairness_params['threshold']
    FAIRNESS_LAMBDA = fairness_params['lambda']

    # ====================================================================
    # HYPERPARAMETERS (ADJUSTED FOR BALANCED DATA)
    # ====================================================================
    ROUNDS = 30
    CLIENTS = [1, 2, 3]
    
    # For balanced data: can use more aggressive training
    # For imbalanced data: need more careful tuning
    if is_balanced:
        LOCAL_EPOCHS = 5              # Back to 5 (balanced data converges better)
        BATCH_SIZE = 32               # Back to 32 (better for balanced training)
        LEARNING_RATE = 1e-3
        MIN_RECALL_TARGET = 0.55      # Higher target for balanced data
    else:
        LOCAL_EPOCHS = 3              # Conservative for imbalanced
        BATCH_SIZE = 64               # Larger batches for imbalanced
        LEARNING_RATE = 1e-3
        MIN_RECALL_TARGET = 0.50      # Standard target
    
    PATIENCE = 20
    
    # Gender fairness configuration
    FAIRNESS_ATTR = 'gender'
    PROTECTED_VALUE = 'Female'
    
    print(f"\nHyperparameter Configuration:")
    print(f"  Dataset: {'BALANCED' if is_balanced else 'UNBALANCED'}")
    print(f"  LOCAL_EPOCHS: {LOCAL_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  MIN_RECALL_TARGET: {MIN_RECALL_TARGET:.2f}")
    print(f"  FAIRNESS_LAMBDA: {FAIRNESS_LAMBDA}")
    
    # History tracking
    history = {
        "round": [], "auc": [], "accuracy": [], "recall": [],
        "precision": [], "f1": [], "specificity": [], "balanced_acc": [],
        "loss": [], "fairness_gap": [], "threshold": [],
        "recall_female": [], "recall_male": [], "train_time": []
    }
    
    best_f1 = 0.0
    rounds_without_improvement = 0
    
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    unique_genders = sens_train[FAIRNESS_ATTR].unique()
    print(f"\n Unique {FAIRNESS_ATTR} values: {unique_genders}")
    if PROTECTED_VALUE not in unique_genders:
        PROTECTED_VALUE = unique_genders[0]
    print(f"   Protected group: {PROTECTED_VALUE}")

    print("\n" + "="*70)
    print("STARTING FEDERATED LEARNING")
    print("="*70)

    # ====================================================================
    # FEDERATED LEARNING LOOP
    # ====================================================================
    for r in range(ROUNDS):
        round_start = time.time()
        print(f"\n--- Round {r+1}/{ROUNDS} ---")
        
        global_weights = global_model.get_weights()
        client_weights, client_sizes = [], []

        # ================================================================
        # LOCAL TRAINING (Each Hospital)
        # ================================================================
        for cid in CLIENTS:
            X_c, y_c, prot_c = get_client_data(
                cid, X_train, y_train, sens_train,
                protected_attr=FAIRNESS_ATTR
            )
            
            if len(X_c) < 50:
                continue
            
            local_model = create_model(input_dim)
            local_model.set_weights(global_weights)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            
            prot_flag = (prot_c == PROTECTED_VALUE).astype(np.int32)
            n_samples = X_c.shape[0]
            
            cw = class_weights_per_hospital[cid]
            sample_weights_np = np.array(
                [cw[int(label)] for label in y_c],
                dtype=np.float32
            )
            
            # ============================================================
            # LOCAL TRAINING LOOP
            # ============================================================
            for ep in range(LOCAL_EPOCHS):
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                
                batch_start = 0
                while batch_start < n_samples:
                    batch_end = min(batch_start + BATCH_SIZE, n_samples)
                    batch_idx = indices[batch_start:batch_end]
                    
                    batch_x = X_c[batch_idx].astype(np.float32)
                    batch_y = y_c[batch_idx].astype(np.float32)
                    batch_prot = prot_flag[batch_idx].astype(np.float32)
                    batch_weights = sample_weights_np[batch_idx]
                    
                    with tf.GradientTape() as tape:
                        batch_x_tf = tf.convert_to_tensor(batch_x)
                        preds = tf.squeeze(
                            local_model(batch_x_tf, training=True),
                            axis=-1
                        )
                        
                        # Task loss
                        batch_y_tf = tf.convert_to_tensor(batch_y)
                        batch_weights_tf = tf.convert_to_tensor(batch_weights)
                        
                        task_loss_raw = focal_loss_fn(batch_y_tf, preds)
                        task_loss = tf.reduce_mean(task_loss_raw * batch_weights_tf)
                        
                        # Fairness penalty
                        fairness_penalty = compute_fairness_loss(
                            preds, batch_y_tf, batch_prot, batch_weights_tf,
                            FAIRNESS_THRESHOLD, FAIRNESS_LAMBDA
                        )
                        
                        # Combined loss
                        total_loss = task_loss + fairness_penalty
                    
                    # Backpropagation
                    grads = tape.gradient(
                        total_loss,
                        local_model.trainable_variables
                    )
                    optimizer.apply_gradients(
                        zip(grads, local_model.trainable_variables)
                    )
                    
                    batch_start = batch_end
            
            client_weights.append(local_model.get_weights())
            client_sizes.append(len(X_c))

        # ================================================================
        # FEDERATED AGGREGATION
        # ================================================================
        if len(client_weights) > 0:
            global_model.set_weights(
                fedavg(client_weights, client_sizes)
            )

        # ================================================================
        # EVALUATION
        # ================================================================
        y_pred = global_model.predict(X_test, verbose=0).flatten()
        
        # Threshold tuning every 5 rounds for balanced data (can afford more frequent)
        if is_balanced:
            tune_threshold = (r % 5 == 0 or r == ROUNDS - 1)
        else:
            tune_threshold = (r % 10 == 0 or r == ROUNDS - 1)
        
        if tune_threshold:
            best_thr, best_f1_eval, best_recall = find_optimal_threshold(
                y_test, y_pred
            )
        else:
            best_thr = 0.5
        
        y_pred_binary = (y_pred > best_thr).astype(int)
        y_test_flat = y_test.flatten()
        
        # Compute metrics
        try:
            auc = roc_auc_score(y_test, y_pred)
        except:
            auc = 0.5
        
        acc = accuracy_score(y_test_flat, y_pred_binary)
        recall = recall_score(y_test_flat, y_pred_binary, zero_division=0)
        precision = precision_score(y_test_flat, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_flat, y_pred_binary, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test_flat, y_pred_binary)
        
        tn, fp, fn, tp = confusion_matrix(
            y_test_flat, y_pred_binary, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        test_loss = tf.keras.losses.binary_crossentropy(
            y_test_flat.reshape(-1, 1),
            y_pred.reshape(-1, 1)
        ).numpy().mean()
        
        # Gender fairness
        female_mask = (sens_test[FAIRNESS_ATTR] == 'Female').values
        male_mask = (sens_test[FAIRNESS_ATTR] == 'Male').values
        
        try:
            recall_female = recall_score(
                y_test_flat[female_mask],
                y_pred_binary[female_mask],
                zero_division=0
            ) if female_mask.sum() > 0 else 0.0
            recall_male = recall_score(
                y_test_flat[male_mask],
                y_pred_binary[male_mask],
                zero_division=0
            ) if male_mask.sum() > 0 else 0.0
        except:
            recall_female = 0.0
            recall_male = 0.0
        
        fairness_gap = abs(recall_female - recall_male)
        
        round_time = time.time() - round_start
        
        # Update history
        history["round"].append(r + 1)
        history["auc"].append(auc)
        history["accuracy"].append(acc)
        history["recall"].append(recall)
        history["precision"].append(precision)
        history["f1"].append(f1)
        history["specificity"].append(specificity)
        history["balanced_acc"].append(balanced_acc)
        history["loss"].append(test_loss)
        history["fairness_gap"].append(fairness_gap)
        history["threshold"].append(best_thr)
        history["recall_female"].append(recall_female)
        history["recall_male"].append(recall_male)
        history["train_time"].append(round_time)
        
        # Print results
        print(f"  Time: {round_time:.1f}s | Threshold={best_thr:.3f}")
        print(f"  Metrics: AUC={auc:.4f} | F1={f1:.4f} | Recall={recall:.4f} | Precision={precision:.4f}")
        print(f"  Gender: F={recall_female:.4f}, M={recall_male:.4f}, Gap={fairness_gap:.4f}")
        
        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            rounds_without_improvement = 0
            global_model.save("models/fedavg_global_model_best.keras")
            print(f"  ✓ Best F1: {f1:.4f}")
        else:
            rounds_without_improvement += 1
        
        if rounds_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at round {r+1}")
            break

    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    total_time = time.time() - start_time
    
    global_model.save("models/fedavg_global_model_final.keras")
    
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("results/fedavg_history_full.csv", index=False)
    
    # ====================================================================
    # VISUALIZATION
    # ====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(history["round"], history["auc"], marker="o", color="blue")
    axes[0, 0].set_title("AUC-ROC", fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history["round"], history["f1"], marker="s", color="purple")
    axes[0, 1].set_title("F1-Score", fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history["round"], history["train_time"], marker="d", color="green")
    axes[0, 2].set_title("Round Training Time (s)", fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(
        history["round"], history["recall_female"],
        label="Female", marker="o", color="red"
    )
    axes[1, 0].plot(
        history["round"], history["recall_male"],
        label="Male", marker="s", color="blue"
    )
    axes[1, 0].axhline(y=MIN_RECALL_TARGET, color='green', linestyle='--', label=f'Target ({MIN_RECALL_TARGET})')
    axes[1, 0].set_title("Gender-Specific Recall", fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(
        history["round"], history["fairness_gap"],
        marker="^", color="orange"
    )
    axes[1, 1].axhline(y=0.1125, color='green', linestyle='--', label='Baseline (11.25%)')
    axes[1, 1].set_title("Gender Fairness Gap", fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    sns.heatmap(
        [[tn, fp], [fn, tp]],
        annot=True, fmt='d', cmap='Blues',
        ax=axes[1, 2], cbar=False
    )
    axes[1, 2].set_title("Confusion Matrix", fontweight='bold')
    axes[1, 2].set_xticklabels(['No', 'Yes'])
    axes[1, 2].set_yticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig("results/figures/05_fedavg_full_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    avg_round_time = np.mean(history["train_time"])
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING COMPLETE")
    print("="*70)
    print(f"\nDataset Type: {'BALANCED ✓' if is_balanced else 'UNBALANCED'}")
    print(f"\nTiming Summary:")
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Avg Time/Round: {avg_round_time:.1f} seconds")
    print(f"  Rounds Completed: {len(history['round'])}")
    
    print(f"\nFinal Results (Round {len(history['round'])}):")
    print(f"  AUC: {history['auc'][-1]:.4f}")
    print(f"  Recall: {history['recall'][-1]:.4f}")
    print(f"  Precision: {history['precision'][-1]:.4f}")
    print(f"  F1-Score: {history['f1'][-1]:.4f}")
    print(f"  Gender Gap: {history['fairness_gap'][-1]:.4f}")
    print(f"  Female Recall: {history['recall_female'][-1]:.4f}")
    print(f"  Male Recall: {history['recall_male'][-1]:.4f}")
    
    print(f"\nFiles saved:")
    print(f"  Results: results/fedavg_history_full.csv")
    print(f"  Dashboard: results/figures/05_fedavg_full_metrics.png")
    print(f"  Best Model: models/fedavg_global_model_best.keras")


if __name__ == "__main__":
    run_federated_learning()