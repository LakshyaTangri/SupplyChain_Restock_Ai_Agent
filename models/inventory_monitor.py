import numpy as np
import pandas as pd
from typing import Dict
import logging
class InventoryMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.safety_stock_model = None
        self.stockout_predictor = None
        self.logger = logging.getLogger(__name__)
    
    def calculate_safety_stock(self, demand_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate safety stock levels based on demand variability."""
        z_score = 1.96  # 95% service level
        results = pd.DataFrame()
        
        for product in demand_data['product_id'].unique():
            product_demand = demand_data[demand_data['product_id'] == product]['units_sold']
            
            lead_time = self.config['lead_time']
            demand_std = product_demand.std()
            demand_mean = product_demand.mean()
            
            safety_stock = z_score * demand_std * np.sqrt(lead_time)
            results = results.append({
                'product_id': product,
                'safety_stock': safety_stock,
                'reorder_point': demand_mean * lead_time + safety_stock
            }, ignore_index=True)
        
        return results