import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import copy
import os

class FederatedClient:
    def __init__(self, client_id, component_type="generic"):
        self.client_id = client_id
        self.component = component_type
        # Default path (can be changed in load_data for other components)
        self.data_path = f"datasets/diabetes_130/processed/client_{client_id}"
        self.model = None
        self.train_loader = None
        
    def load_data(self, data_client_id=None):
        """
        Loads data dynamically based on the component type.
        This function acts as a 'Switch' to handle different team members' datasets.
        
        :param data_client_id: If provided, loads data from this ID instead of self.client_id.
                               Useful for scalability testing (reusing data).
        """
        # Determine which file to load. 
        # If data_client_id is passed (e.g. 0), we load client_0 data even if self.client_id is 5.
        file_id = data_client_id if data_client_id is not None else self.client_id
        
        # Update path to point to the correct file
        target_data_path = f"datasets/diabetes_130/processed/client_{file_id}"
        
        # Update self.data_path so fairness checks look at the right file too
        self.data_path = target_data_path

        print(f"Client {self.client_id}: Loading data for {self.component} (Source: Client {file_id})...")
        dataset = None

        # --- OPTION A: Member 1 & 4 -> Diabetes 130 Dataset ---
        # Used by Component 2 (Readmission) & Component 4 (Multi-Task)
        if self.component in ["comp2_readmission", "comp4_multitask", "comp4_singletask"]:
            try:
                # Check if files exist
                if not os.path.exists(f"{target_data_path}_X.csv"):
                    print(f"❌ Error: Data file not found: {target_data_path}_X.csv")
                    print("   -> Run 'python datasets/diabetes_130/preprocess.py'?")
                    # Initialize empty so we don't crash, but won't train
                    self.train_loader = []
                    return

                X = pd.read_csv(f"{target_data_path}_X.csv").values
                y = pd.read_csv(f"{target_data_path}_y.csv")
                
                # Select Targets based on Component
                if self.component == "comp2_readmission":
                    # Component 2 only cares about Readmission (Column 0)
                    y_target = y['target_readmission'].values
                elif self.component == "comp4_singletask":
                 # Control Experiment: Only predict Hypertension (Column 0)
                 y_target = y['target_hypertension'].values
                else:
                    # Component 4 uses the first two columns (Hypertension, Heart Failure)
                    y_target = y[['target_hypertension', 'target_heart_failure']].values

                # Create PyTorch Tensors
                tensor_x = torch.Tensor(X)
                tensor_y = torch.Tensor(y_target)
                dataset = TensorDataset(tensor_x, tensor_y)
                
            except Exception as e:
                print(f"❌ Error loading Diabetes 130 data: {e}")
                self.train_loader = []
                return

        # --- OPTION B: Member 3 (Images/Multimodal) ---
        elif self.component == "comp3_multimodal":
            # from components.component_3.dataset import ImageDataset
            # dataset = ImageDataset(root_dir="datasets/multimodal/images", client_id=self.client_id)
            print("⚠️ Client needs to implement ImageDataset loading logic in fl_core/client.py")
            return 

        # --- OPTION C: Member 1 (Privacy/Complications) ---
        elif self.component == "comp1_privacy":
            # If using a different dataset, load it here.
            print("⚠️ Client needs to implement Privacy Dataset loading logic in fl_core/client.py")
            return

        else:
            print(f"❌ Unknown component type: {self.component}")
            return

        # Finalize the Loader
        if dataset:
            self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            print(f"   -> Data loaded successfully. ({len(dataset)} samples)")
        else:
            self.train_loader = [] # Safety fallback

    def set_model(self, model):
        """Receives the global model from the server."""
        self.model = model

    def train(self, epochs=1):
        """Local training loop."""
        if not self.model:
            raise ValueError("Model not set!")
        
        # Check if we have data
        if not self.train_loader or len(self.train_loader) == 0:
            # print(f"Client {self.client_id}: No data to train on. Skipping.")
            return self.model.state_dict() # Return unchanged weights

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss() # Binary Cross Entropy
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                # --- MULTI-TASK LEARNING (Component 4) ---
                if self.component == "comp4_multitask":
                    # The model returns TWO outputs
                    pred_htn, pred_hf = self.model(batch_X)
                    
                    # The target batch also has two columns
                    target_htn = batch_y[:, 0].unsqueeze(1)
                    target_hf = batch_y[:, 1].unsqueeze(1)
                    
                    # Calculate loss for both tasks and add them together
                    loss_htn = criterion(pred_htn, target_htn)
                    loss_hf = criterion(pred_hf, target_hf)
                    loss = loss_htn + loss_hf
                    
                # --- GENERIC COMPONENTS (Comp 1, 2 & 4 Single-Task) ---
                else:
                    outputs = self.model(batch_X)
                    # Reshape batch_y from [32] to [32, 1] to match outputs
                    loss = criterion(outputs, batch_y.view(-1, 1))

                loss.backward()
                optimizer.step()
        
        # print(f"Client {self.client_id}: Training complete.")
        return self.model.state_dict() # Return weights to server

    def evaluate_personalization(self, epochs=10):
        """
        Research Objective 2.1 & 2.2:
        Fine-tunes the global model on local data to measure Personalization Gain.
        Returns: Accuracy BEFORE fine-tuning vs. Accuracy AFTER fine-tuning.
        Research Objective 2.1: Adaptive Parameter Sharing.
        Freezes the 'Shared Body' and only fine-tunes the 'Task Heads'.
        """
        if not self.model:
            return 0.0, 0.0
        
        # Check for data
        if not self.train_loader or len(self.train_loader) == 0:
            return 0.0, 0.0

        # 1. Measure Baseline (Global Model Performance)
        baseline_acc = self._calculate_accuracy()
        
        # 2. Personalization Step (Fine-Tuning)
        # Create a COPY so we don't mess up the main training loop
        personalized_model = copy.deepcopy(self.model)
        
        # === THE "ADAPTIVE" LOGIC ===
        # We iterate through layers. If it's a "shared" layer, we freeze it.
        # If it's a "head" layer, we let it learn.
        for name, param in personalized_model.named_parameters():
            if "shared" in name or "bn" in name: # Freeze Body & Batch Norm
                param.requires_grad = False
            else: # Train Heads
                param.requires_grad = True
        
        # Only optimize parameters that require gradient
        # Add L2 Regularization (Weight Decay) to prevent overfitting during personalization
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, personalized_model.parameters()), 
            lr=0.001, 
            weight_decay=1e-5  # <--- THIS ADDS L2 REGULARIZATION
        )
        criterion = torch.nn.BCELoss()
        
        # 3. Training Loop (Standard)
        personalized_model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                if self.component == "comp4_multitask":
                    pred_htn, pred_hf = personalized_model(batch_X)
                    loss = criterion(pred_htn, batch_y[:, 0].unsqueeze(1)) + \
                           criterion(pred_hf, batch_y[:, 1].unsqueeze(1))
                else:
                    pred = personalized_model(batch_X)
                    loss = criterion(pred, batch_y.view(-1, 1))
                    
                loss.backward()
                optimizer.step()
                
        # 4. Measure Personalized Performance
        # Swap models temporarily
        temp_model = self.model 
        self.model = personalized_model 
        personalized_acc = self._calculate_accuracy()
        self.model = temp_model # Swap back
        
        # print(f"Client {self.client_id} Personalization Gain: {personalized_acc - baseline_acc:.4f}")
        return baseline_acc, personalized_acc

    def _calculate_accuracy(self):
        """Helper to calculate accuracy on local data."""
        if not self.train_loader or len(self.train_loader) == 0:
            return 0.0

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_y in self.train_loader:
                if self.component == "comp4_multitask":
                    p_htn, p_hf = self.model(batch_X)
                    # Average accuracy of both tasks
                    acc_htn = ((p_htn > 0.5) == (batch_y[:, 0].unsqueeze(1) > 0.5)).sum().item()
                    acc_hf = ((p_hf > 0.5) == (batch_y[:, 1].unsqueeze(1) > 0.5)).sum().item()
                    correct += (acc_htn + acc_hf) / 2 # Avg correct
                else:
                    outputs = self.model(batch_X)
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == batch_y.view(-1, 1)).sum().item()
                total += batch_y.size(0)
        
        if total == 0: return 0.0
        return correct / total
    
    def evaluate_fairness(self):
        """
        Research Objective 2.2.iii & Target 4:
        Calculate Demographic Parity Gap (Fairness).
        Handles Standardized Data (where values are not exactly 0 or 1).
        """
        if not self.model: return 0.0
        
        try:
             # We try to read the file currently assigned to this client.
             if not os.path.exists(f"{self.data_path}_X.csv"):
                 return 0.0
             else:
                df = pd.read_csv(f"{self.data_path}_X.csv")
                y = pd.read_csv(f"{self.data_path}_y.csv")
        except:
            return 0.0
        
        # --- SMART COLUMN DETECTION (SCALING PROOF) ---
        group_a = None # Females
        group_b = None # Males

        # We look for the column. Since data is scaled, we use > 0 or < 0
        if 'gender_Female' in df.columns:
            # If Scaled: > 0 is likely 1 (Female), < 0 is likely 0 (Male)
            group_a = df[df['gender_Female'] > 0] # Females
            group_b = df[df['gender_Female'] < 0] # Males
        elif 'gender_Male' in df.columns:
            # If Scaled: < 0 is likely 0 (Female), > 0 is likely 1 (Male)
            group_a = df[df['gender_Male'] < 0] # Females
            group_b = df[df['gender_Male'] > 0] # Males
        else:
            return 0.0

        # Helper to get accuracy
        def get_acc(subset_X, subset_indices):
            if len(subset_X) == 0: return 0.0
            subset_y = y.iloc[subset_indices]
            t_X = torch.Tensor(subset_X.values)
            
            self.model.eval()
            with torch.no_grad():
                if self.component == "comp4_multitask":
                    p_htn, p_hf = self.model(t_X)
                    preds = (p_htn > 0.5).float().numpy()
                    targets = subset_y['target_hypertension'].values.reshape(-1, 1)
                else:
                    out = self.model(t_X)
                    preds = (out > 0.5).float().numpy()
                    targets = subset_y.iloc[:, 0].values.reshape(-1, 1)
            
            return (preds == targets).mean()

        # Calculate Accuracy
        acc_female = get_acc(group_a, group_a.index)
        acc_male = get_acc(group_b, group_b.index)
        
        gap = abs(acc_female - acc_male)
        
        return gap