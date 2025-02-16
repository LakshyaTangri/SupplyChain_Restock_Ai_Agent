from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.train_size = config.get('train_size', 0.7)
        self.test_size = config.get('test_size', 0.15)
        self.val_size = config.get('val_size', 0.15)

    def process_upload(self, df: pd.DataFrame) -> Dict:
        """Process uploaded data and return dashboard metrics."""
        
        # Perform train/test/validation split
        train_data, test_val_data = train_test_split(
            df, 
            train_size=self.train_size,
            random_state=42
        )
        
        test_data, val_data = train_test_split(
            test_val_data,
            test_size=self.val_size/(self.test_size + self.val_size),
            random_state=42
        )

        # Generate metrics and predictions for each model
        demand_metrics, demand_predictions = self._get_demand_metrics(train_data, test_data)
        inventory_metrics, inventory_predictions = self._get_inventory_metrics(train_data, test_data)
        price_metrics, price_predictions = self._get_price_metrics(train_data, test_data)

        return {
            'splits': {
                'train': len(train_data),
                'test': len(test_data),
                'validation': len(val_data)
            },
            'demandPredictions': self._format_predictions(demand_predictions),
            'inventoryPredictions': self._format_predictions(inventory_predictions),
            'pricePredictions': self._format_predictions(price_predictions),
            'metrics': {
                'demand': demand_metrics,
                'inventory': inventory_metrics,
                'price': price_metrics
            }
        }

    def _get_demand_metrics(self, train_data: pd.DataFrame, 
                           test_data: pd.DataFrame) -> Tuple[Dict, List[Dict]]:
        """Calculate demand forecasting metrics and predictions."""
        # Simulate model predictions for demonstration
        predictions = test_data['units_sold'].values
        actuals = test_data['units_sold'].values
        
        # Add some random variation for demonstration
        predictions = predictions * (1 + np.random.normal(0, 0.1, len(predictions)))
        
        metrics = self._calculate_metrics(actuals, predictions)
        
        chart_data = []
        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            chart_data.append({
                'date': test_data.iloc[i]['date'],
                'predicted': pred,
                'actual': actual
            })
        
        return metrics, chart_data

    def _get_inventory_metrics(self, train_data: pd.DataFrame, 
                             test_data: pd.DataFrame) -> Tuple[Dict, List[Dict]]:
        """Calculate inventory optimization metrics and predictions."""
        predictions = test_data['inventory_level'].values
        actuals = test_data['inventory_level'].values
        
        predictions = predictions * (1 + np.random.normal(0, 0.1, len(predictions)))
        
        metrics = self._calculate_metrics(actuals, predictions)
        
        chart_data = []
        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            chart_data.append({
                'date': test_data.iloc[i]['date'],
                'predicted': pred,
                'actual': actual
            })
        
        return metrics, chart_data

    def _get_price_metrics(self, train_data: pd.DataFrame, 
                          test_data: pd.DataFrame) -> Tuple[Dict, List[Dict]]:
        """Calculate price optimization metrics and predictions."""
        predictions = test_data['price'].values
        actuals = test_data['price'].values
        
        predictions = predictions * (1 + np.random.normal(0, 0.05, len(predictions)))
        
        metrics = self._calculate_metrics(actuals, predictions)
        
        chart_data = []
        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            chart_data.append({
                'date': test_data.iloc[i]['date'],
                'predicted': pred,
                'actual': actual
            })
        
        return metrics, chart_data

    def _calculate_metrics(self, actuals: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate performance metrics."""
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # Calculate R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    def _format_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Format predictions for dashboard charts."""
        return [
            {
                'date': pred['date'],
                'predicted': float(pred['predicted']),
                'actual': float(pred['actual'])
            }
            for pred in predictions
        ]