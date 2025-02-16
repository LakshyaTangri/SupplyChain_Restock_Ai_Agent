class SupplierRatingModel(nn.Module):
    def __init__(self, n_suppliers: int, n_features: int, embedding_dim: int = 50):
        super().__init__()
        self.supplier_embeddings = nn.Embedding(n_suppliers, embedding_dim)
        self.feature_network = nn.Sequential(
            nn.Linear(n_features + embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, supplier_ids, features):
        supplier_emb = self.supplier_embeddings(supplier_ids)
        combined = torch.cat([supplier_emb, features], dim=1)
        return self.feature_network(combined)

class SupplierManager:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Train the supplier rating model."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_suppliers, batch_features, batch_ratings in train_loader:
                batch_suppliers = batch_suppliers.to(self.device)
                batch_features = batch_features.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_suppliers, batch_features)
                loss = criterion(outputs, batch_ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            self.logger.info(f'Epoch {epoch+1}/{epochs}')
            self.logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}')