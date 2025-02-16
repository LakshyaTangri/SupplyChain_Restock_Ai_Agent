import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
import logging
from fastai.tabular.all import *

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=0.2,
                           batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

class DemandForecaster:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    def build_model(self, input_size: int):
        """Initialize the LSTM model."""
        self.model = LSTMForecaster(input_size=input_size,
                                  hidden_size=self.config['hidden_size'],
                                  num_layers=self.config['num_layers'])
        self.model.to(self.device)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Train the demand forecasting model."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            self.logger.info(f'Epoch {epoch+1}/{epochs}')
            self.logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}')
            self.logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}')
