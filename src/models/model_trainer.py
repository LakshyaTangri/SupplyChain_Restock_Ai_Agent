# src/models/model_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class SupplyChainDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class SupplyChainModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    def __init__(self, input_size, hidden_size=128, learning_rate=0.001):
        self.model = SupplyChainModel(input_size, hidden_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train_model(self, train_loader, val_loader, epochs=100, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for features, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets.view(-1, 1))
                    val_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features, _ in test_loader:
                outputs = self.model(features)
                predictions.extend(outputs.numpy().flatten())
        return np.array(predictions)