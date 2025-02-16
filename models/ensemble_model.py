class EnsembleModel:
    def __init__(self, models: List[nn.Module], weights: Optional[np.ndarray] = None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        self.logger = logging.getLogger(__name__)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Generate ensemble predictions."""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred * weight)
        
        return torch.stack(predictions).sum(dim=0)
    
    def optimize_weights(self, val_loader: DataLoader, metric_func) -> np.ndarray:
        """Optimize ensemble weights using validation data."""
        model_predictions = []
        true_values = []
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            model_preds = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    pred = model(batch_X)
                    model_preds.append(pred)
                    if len(model_predictions) == 0:
                        true_values.append(batch_y)
            
            model_predictions.append(torch.cat(model_preds))
        
        true_values = torch.cat(true_values)
        model_predictions = torch.stack(model_predictions)
        
        # Optimize weights using scipy's optimize
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_pred = (model_predictions * torch.tensor(weights).reshape(-1, 1)).sum(dim=0)
            return metric_func(true_values, ensemble_pred)
        
        initial_weights = np.ones(len(self.models)) / len(self.models)
        bounds = [(0, 1) for _ in range(len(self.models))]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        self.weights = result.x
        
        return self.weights

if __name__ == "__main__":
    # Configuration
    config = {
        'hidden_size': 128,
        'num_layers': 2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'lead_time': 7,
        'inventory_step': 100,
        'demand_step': 50,
        'n_actions': 10
    }
    
    # Initialize models
    demand_forecaster = DemandForecaster(config)
    inventory_monitor = InventoryMonitor(config)
    replenishment_optimizer = ReplenishmentOptimizer(config)
    supplier_manager = SupplierManager(config)
    price_optimizer = PriceOptimizer(config)
    
    # Example ensemble
    models = [demand_forecaster.model, inventory_monitor.stockout_predictor]
    ensemble = EnsembleModel(models)