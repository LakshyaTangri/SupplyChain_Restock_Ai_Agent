# data/schemas/data_schemas.py
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

@dataclass
class SupplyChainSchema:
    """Schema definition for supply chain data."""
    
    COLUMNS: List[str] = [
        'date', 'store_id', 'product_id', 'category', 'region',
        'inventory_level', 'units_sold', 'units_ordered',
        'demand_forecast', 'price', 'discount', 'weather_condition',
        'holiday_promotion', 'competitor_pricing', 'seasonality'
    ]
    
    SCHEMA_RULES: Dict = {
        'date': 'datetime64[ns]',
        'store_id': 'string',
        'product_id': 'string',
        'category': 'category',
        'region': 'category',
        'inventory_level': 'int64',
        'units_sold': 'int64',
        'units_ordered': 'int64',
        'demand_forecast': 'float64',
        'price': 'float64',
        'discount': 'float64',
        'weather_condition': 'category',
        'holiday_promotion': 'int64',
        'competitor_pricing': 'float64',
        'seasonality': 'category'
    }
    
    CATEGORICAL_COLUMNS: List[str] = [
        'category', 'region', 'weather_condition', 'seasonality'
    ]
    
    NUMERICAL_COLUMNS: List[str] = [
        'inventory_level', 'units_sold', 'units_ordered',
        'demand_forecast', 'price', 'discount',
        'holiday_promotion', 'competitor_pricing'
    ]