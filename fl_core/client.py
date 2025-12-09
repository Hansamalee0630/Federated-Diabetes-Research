# # fl_core/client.py (UPDATED VERSION)
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import copy

# class FederatedClient:
#     def __init__(self, client_id, component_type="generic"):
#         self.client_id = client_id
#         self.component = component_type
#         self.data_path = f"datasets/diabetes_130/processed/client_{client_id}"
#         self.model = None
#         self.train_loader = None
        
#     def load_data(self):
#         """Loads data specific to the assigned component task."""
#         print(f"Client {self.client_id}: Loading data...")
#         try:
#             X = pd.read_csv(f"{self.data_path}_X.csv").values
#             y = pd.read_csv(f"{self.data_path}_y.csv")
#         except FileNotFoundError:
#             print(f"Error: Data file not found at {self.data_path}_X.csv")
#             return

#         # SELECT TARGET BASED ON COMPONENT
#         if self.component == "comp2_readmission":
#             # Component 2 only cares about Readmission (Column 0)
#             y_target = y['target_readmission'].values
#         elif self.component == "comp4_multitask":
#             # Component 4 predicts Comorbidities (Columns 1 & 2)
#             y_target = y[['target_hypertension', 'target_heart_failure']].values
#         else:
#             y_target = y.values # Default all

#         # Convert to PyTorch Tensors
#         tensor_x = torch.Tensor(X)
#         tensor_y = torch.Tensor(y_target)
        
#         dataset = TensorDataset(tensor_x, tensor_y)
#         self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#         print(f"Client {self.client_id}: Data loaded for {self.component}. Batches: {len(self.train_loader)}")

#     def set_model(self, model):
#         """Receives the global model from the server."""
#         self.model = model

#     def train(self, epochs=1):
#         """Local training loop."""
#         if not self.model:
#             raise ValueError("Model not set!")
        
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
#         criterion = torch.nn.BCELoss() # Binary Cross Entropy
        
#         self.model.train()
#         for epoch in range(epochs):
#             for batch_X, batch_y in self.train_loader:
#                 optimizer.zero_grad()
                
#                 # --- NEW LOGIC FOR MULTI-TASK LEARNING ---
#                 if self.component == "comp4_multitask":
#                     # The model returns TWO outputs (Hypertension, Heart Failure)
#                     pred_htn, pred_hf = self.model(batch_X)
                    
#                     # The target batch also has two columns
#                     target_htn = batch_y[:, 0].unsqueeze(1)
#                     target_hf = batch_y[:, 1].unsqueeze(1)
                    
#                     # Calculate loss for both tasks and add them together
#                     loss_htn = criterion(pred_htn, target_htn)
#                     loss_hf = criterion(pred_hf, target_hf)
#                     loss = loss_htn + loss_hf
                    
#                 else:
#                     # --- GENERIC LOGIC FOR OTHER COMPONENTS ---
#                     outputs = self.model(batch_X) # Shape: [32, 1]
                    
#                     # === THE FIX IS HERE ===
#                     # Reshape batch_y from [32] to [32, 1] to match outputs
#                     loss = criterion(outputs, batch_y.view(-1, 1))

#                 loss.backward()
#                 optimizer.step()
        
#         print(f"Client {self.client_id}: Training complete.")
#         return self.model.state_dict() # Return weights to server
    
#     # ... inside fl_core/client.py ...

#     def evaluate_personalization(self, epochs=5):
#         """
#         Research Objective 2.1 & 2.2:
#         Fine-tunes the global model on local data to measure Personalization Gain.
#         Returns: Accuracy BEFORE fine-tuning vs. Accuracy AFTER fine-tuning.
#         """
#         if not self.model:
#             return 0.0, 0.0
            
#         # 1. Measure Baseline (Global Model Performance)
#         # We test on the TRAINING set here just to see adaptation capability
#         # In real research, you'd split local data into Local-Train and Local-Test
#         baseline_acc = self._calculate_accuracy()
        
#         # 2. Personalization Step (Fine-Tuning)
#         # We create a COPY so we don't mess up the main training loop
#         personalized_model = copy.deepcopy(self.model)
#         optimizer = torch.optim.Adam(personalized_model.parameters(), lr=0.001)
#         criterion = torch.nn.BCELoss()
        
#         personalized_model.train()
#         for epoch in range(epochs):
#             for batch_X, batch_y in self.train_loader:
#                 optimizer.zero_grad()
#                 if self.component == "comp4_multitask":
#                     pred_htn, pred_hf = personalized_model(batch_X)
#                     loss = criterion(pred_htn, batch_y[:, 0].unsqueeze(1)) + \
#                            criterion(pred_hf, batch_y[:, 1].unsqueeze(1))
#                 else:
#                     pred = personalized_model(batch_X)
#                     loss = criterion(pred, batch_y.view(-1, 1))
#                 loss.backward()
#                 optimizer.step()
                
#         # 3. Measure Personalized Performance
#         # Use the personalized model to check accuracy
#         temp_model = self.model # Save global
#         self.model = personalized_model # Swap in personal
#         personalized_acc = self._calculate_accuracy()
#         self.model = temp_model # Swap back global
        
#         print(f"Client {self.client_id} Personalization Gain: {personalized_acc - baseline_acc:.4f}")
#         return baseline_acc, personalized_acc

#     def _calculate_accuracy(self):
#         """Helper to calculate accuracy on local data."""
#         correct = 0
#         total = 0
#         self.model.eval()
#         with torch.no_grad():
#             for batch_X, batch_y in self.train_loader:
#                 if self.component == "comp4_multitask":
#                     p_htn, p_hf = self.model(batch_X)
#                     # Average accuracy of both tasks
#                     acc_htn = ((p_htn > 0.5) == (batch_y[:, 0].unsqueeze(1) > 0.5)).sum().item()
#                     acc_hf = ((p_hf > 0.5) == (batch_y[:, 1].unsqueeze(1) > 0.5)).sum().item()
#                     correct += (acc_htn + acc_hf) / 2 # Avg correct
#                 else:
#                     outputs = self.model(batch_X)
#                     predicted = (outputs > 0.5).float()
#                     correct += (predicted == batch_y.view(-1, 1)).sum().item()
#                 total += batch_y.size(0)
#         return correct / 

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
        
    def load_data(self):
        """
        Loads data dynamically based on the component type.
        This function acts as a 'Switch' to handle different team members' datasets.
        """
        print(f"Client {self.client_id}: Loading data for {self.component}...")
        dataset = None

        # --- OPTION A: Your Group (Diabetes 130 Dataset) ---
        # Used by Component 2 (Readmission) & Component 4 (Multi-Task)
        if self.component in ["comp2_readmission", "comp4_multitask"]:
            try:
                # Check if files exist
                if not os.path.exists(f"{self.data_path}_X.csv"):
                    print(f"❌ Error: Data file not found: {self.data_path}_X.csv")
                    print("   -> Did you run 'python datasets/diabetes_130/preprocess.py'?")
                    return

                X = pd.read_csv(f"{self.data_path}_X.csv").values
                y = pd.read_csv(f"{self.data_path}_y.csv")
                
                # Select Targets based on Component
                if self.component == "comp2_readmission":
                    # Component 2 only cares about Readmission (Column 0)
                    y_target = y['target_readmission'].values
                else:
                    # Component 4 uses the first two columns (Hypertension, Heart Failure)
                    y_target = y[['target_hypertension', 'target_heart_failure']].values

                # Create PyTorch Tensors
                tensor_x = torch.Tensor(X)
                tensor_y = torch.Tensor(y_target)
                dataset = TensorDataset(tensor_x, tensor_y)
                
            except Exception as e:
                print(f"❌ Error loading Diabetes 130 data: {e}")
                return

        # --- OPTION B: Member 3 (Images/Multimodal) ---
        elif self.component == "comp3_multimodal":
            # [INSTRUCTION FOR MEMBER 3]
            # Write your image loading logic here.
            # Example:
            # from components.component_3.dataset import ImageDataset
            # dataset = ImageDataset(root_dir="datasets/multimodal/images", client_id=self.client_id)
            print("⚠️ Client needs to implement ImageDataset loading logic in fl_core/client.py")
            return 

        # --- OPTION C: Member 1 (Privacy/Complications) ---
        elif self.component == "comp1_privacy":
            # [INSTRUCTION FOR MEMBER 1]
            # If using a different dataset, load it here.
            print("⚠️ Client needs to implement Privacy Dataset loading logic in fl_core/client.py")
            return

        else:
            print(f"❌ Unknown component type: {self.component}")
            return

        # Finalize the Loader
        if dataset:
            self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            print(f"Client {self.client_id}: Data loaded successfully. ({len(dataset)} samples)")

    def set_model(self, model):
        """Receives the global model from the server."""
        self.model = model

    def train(self, epochs=1):
        """Local training loop."""
        if not self.model:
            raise ValueError("Model not set!")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss() # Binary Cross Entropy
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                # --- LOGIC FOR MULTI-TASK LEARNING (Component 4) ---
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
                    
                # --- LOGIC FOR GENERIC COMPONENTS (Comp 1 & 2) ---
                else:
                    outputs = self.model(batch_X)
                    # Reshape batch_y from [32] to [32, 1] to match outputs
                    loss = criterion(outputs, batch_y.view(-1, 1))

                loss.backward()
                optimizer.step()
        
        print(f"Client {self.client_id}: Training complete.")
        return self.model.state_dict() # Return weights to server

    def evaluate_personalization(self, epochs=10):
        """
        Research Objective 2.1 & 2.2:
        Fine-tunes the global model on local data to measure Personalization Gain.
        Returns: Accuracy BEFORE fine-tuning vs. Accuracy AFTER fine-tuning.
        """
        if not self.model:
            return 0.0, 0.0
            
        # 1. Measure Baseline (Global Model Performance)
        baseline_acc = self._calculate_accuracy()
        
        # 2. Personalization Step (Fine-Tuning)
        # Create a COPY so we don't mess up the main training loop
        personalized_model = copy.deepcopy(self.model)
        optimizer = torch.optim.Adam(personalized_model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
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
                
        # 3. Measure Personalized Performance
        # Swap models temporarily
        temp_model = self.model 
        self.model = personalized_model 
        personalized_acc = self._calculate_accuracy()
        self.model = temp_model # Swap back
        
        print(f"Client {self.client_id} Personalization Gain: {personalized_acc - baseline_acc:.4f}")
        return baseline_acc, personalized_acc

    def _calculate_accuracy(self):
        """Helper to calculate accuracy on local data."""
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
        return correct / total
    
    def evaluate_fairness(self):
        """
        Research Objective 2.2.iii & Target 4:
        Calculate Demographic Parity Gap (Fairness).
        Handles Standardized Data (where values are not exactly 0 or 1).
        """
        if not self.model: return {}
        
        # Load data
        try:
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
            # Fallback: Check for other common columns if needed
            print(f"⚠️ Fairness feature (gender) not found in {list(df.columns[:5])}...")
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
        
        print(f"⚖️ FAIRNESS CHECK (Client {self.client_id}):")
        print(f"   Female Acc: {acc_female:.4f} | Male Acc: {acc_male:.4f}")
        print(f"   Gap: {gap:.4f} (Target <= 0.05)")
        
        return gap