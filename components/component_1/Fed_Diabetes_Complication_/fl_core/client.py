"""import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
import warnings

# Suppress Opacus hooks warnings for cleaner terminal output in research
warnings.filterwarnings("ignore", message="Full backward hook is firing")

class DiabetesClient:
    def __init__(self, client_id, model, train_loader):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # reduction='none' is mandatory for Opacus to calculate per-sample gradients
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)

        # Initialize Privacy Engine
        # Note: For your FINAL paper run, you can set secure_mode=True
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model, 
            optimizer=self.optimizer, 
            data_loader=self.train_loader,
            noise_multiplier=0.5, 
            max_grad_norm=1.0,
        )

    def local_train(self, global_weights):
        # Map global weights to the Opacus-wrapped module
        wrapped_weights = {"_module." + k: v for k, v in global_weights.items()}
        self.model.load_state_dict(wrapped_weights, strict=False)
        self.model.train()
        
        batch_losses = []
        for inputs, labels, weights in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).view(-1, 1).float()
            weights = weights.to(self.device).view(-1, 1).float()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # 1. Calculate per-sample loss
            raw_loss = self.criterion(outputs, labels)
            
            # 2. Apply weights (this handles class imbalance)
            # We multiply by weights before taking the mean so Opacus sees 
            # the scaled importance of each sample's gradient.
            weighted_batch_loss = (raw_loss * weights).mean() 
            
            # 3. Backward pass
            weighted_batch_loss.backward()
            self.optimizer.step()
            
            batch_losses.append(weighted_batch_loss.item())
        
        avg_loss = np.mean(batch_losses) if batch_losses else 0
        
        # Extract weights and remove the "_module." prefix for the server
        state_dict = self.model.state_dict()
        clean_weights = {k.replace("_module.", ""): v for k, v in state_dict.items()}
        
        return clean_weights, avg_loss """

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
import warnings

# Suppress Opacus hooks warnings for cleaner terminal output
warnings.filterwarnings("ignore", message="Full backward hook is firing")

class DiabetesClient:
    def __init__(self, client_id, model, train_loader):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Mandatory 'none' reduction for Opacus per-sample gradients
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)

        # Initialize Privacy Engine
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model, 
            optimizer=self.optimizer, 
            data_loader=self.train_loader,
            noise_multiplier=0.5, 
            max_grad_norm=1.0,
        )

    def local_train(self, global_weights, epochs=3):
        # 1. Sync weights from server
        wrapped_weights = {"_module." + k: v for k, v in global_weights.items()}
        self.model.load_state_dict(wrapped_weights, strict=False)
        self.model.train()
        
        all_epoch_losses = []
        
        # 2. NEW: Multi-epoch local loop
        for epoch in range(epochs):
            batch_losses = []
            for inputs, labels, weights in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).view(-1, 1).float()
                weights = weights.to(self.device).view(-1, 1).float()
                
                self.optimizer.zero_grad()
                
                # Forward pass (Uses new 32-neuron + Dropout architecture)
                outputs = self.model(inputs)
                
                # Per-sample loss scaled by fuzzy clinical weights
                raw_loss = self.criterion(outputs, labels)
                weighted_batch_loss = (raw_loss * weights).mean() 
                
                # Backward pass with Differential Privacy
                weighted_batch_loss.backward()
                self.optimizer.step()
                
                batch_losses.append(weighted_batch_loss.item())
            
            epoch_avg = np.mean(batch_losses) if batch_losses else 0
            all_epoch_losses.append(epoch_avg)
            # print(f"Client {self.client_id} | Epoch {epoch+1}/{epochs} | Loss: {epoch_avg:.4f}")

        # 3. Cleanup and Return
        avg_loss = np.mean(all_epoch_losses)
        state_dict = self.model.state_dict()
        clean_weights = {k.replace("_module.", ""): v for k, v in state_dict.items()}
        
        return clean_weights, avg_loss
        