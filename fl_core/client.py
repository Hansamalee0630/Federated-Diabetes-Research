# fl_core/client.py (UPDATED VERSION)
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

class FederatedClient:
    def __init__(self, client_id, component_type="generic"):
        self.client_id = client_id
        self.component = component_type
        self.data_path = f"datasets/diabetes_130/processed/client_{client_id}"
        self.model = None
        self.train_loader = None
        
    def load_data(self):
        """Loads data specific to the assigned component task."""
        print(f"Client {self.client_id}: Loading data...")
        try:
            X = pd.read_csv(f"{self.data_path}_X.csv").values
            y = pd.read_csv(f"{self.data_path}_y.csv")
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}_X.csv")
            return

        # SELECT TARGET BASED ON COMPONENT
        if self.component == "comp2_readmission":
            # Component 2 only cares about Readmission (Column 0)
            y_target = y['target_readmission'].values
        elif self.component == "comp4_multitask":
            # Component 4 predicts Comorbidities (Columns 1 & 2)
            y_target = y[['target_hypertension', 'target_heart_failure']].values
        else:
            y_target = y.values # Default all

        # Convert to PyTorch Tensors
        tensor_x = torch.Tensor(X)
        tensor_y = torch.Tensor(y_target)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"Client {self.client_id}: Data loaded for {self.component}. Batches: {len(self.train_loader)}")

    def set_model(self, model):
        """Receives the global model from the server."""
        self.model = model

    def train(self, epochs=1):
        """Local training loop."""
        if not self.model:
            raise ValueError("Model not set!")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss() # Binary Cross Entropy for prediction
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                
                # --- NEW LOGIC FOR MULTI-TASK LEARNING ---
                if self.component == "comp4_multitask":
                    # The model returns TWO outputs (Hypertension, Heart Failure)
                    pred_htn, pred_hf = self.model(batch_X)
                    
                    # The target batch also has two columns
                    target_htn = batch_y[:, 0].unsqueeze(1)
                    target_hf = batch_y[:, 1].unsqueeze(1)
                    
                    # Calculate loss for both tasks and add them together
                    loss_htn = criterion(pred_htn, target_htn)
                    loss_hf = criterion(pred_hf, target_hf)
                    loss = loss_htn + loss_hf
                    
                else:
                    # --- GENERIC LOGIC FOR OTHER COMPONENTS ---
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
        
        print(f"Client {self.client_id}: Training complete.")
        return self.model.state_dict() # Return weights to server