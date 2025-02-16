# src/utils/data_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

data = pd.read_csv('data/retail_store_inventory.csv')
df = pd.DataFrame(data)

class SupplyChainDataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def process_date(self, df):
        """Extract temporal features from date column."""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfMonth'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        return df
    
    def encode_categorical(self, df, columns=['Store ID', 'Product ID', 'Category', 
                                            'Region', 'Weather Condition']):
        """Encode categorical variables."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_Encoded'] = self.encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_Encoded'] = self.encoders[col].transform(df[col])
        return df
    
    def scale_numerical(self, df, columns=['Inventory Level', 'Units Sold', 'Units Ordered',
                                         'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']):
        """Scale numerical features."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[f'{col}_Scaled'] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df[f'{col}_Scaled'] = self.scalers[col].transform(df[[col]])
        return df
    
    def create_binary_features(self, df):
        """Create binary features for holidays/promotions."""
        df = df.copy()
        df['HasPromotion'] = df['Holiday/Promotion'].notna().astype(int)
        return df
    
    def create_lag_features(self, df, columns=['Units Sold', 'Price'], lags=[1, 7, 14]):
        """Create lag features for specified columns."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_Lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])[col].shift(lag)
        return df
    
    def create_rolling_features(self, df, columns=['Units Sold'], windows=[7, 14, 30]):
        """Create rolling mean features."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_Rolling_Mean_{window}'] = df.groupby(['Store ID', 'Product ID'])[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean())
        return df
    
    def process_data(self, df, create_lags=True):
        """Complete data processing pipeline."""
        df = self.process_date(df)
        df = self.encode_categorical(df)
        df = self.scale_numerical(df)
        df = self.create_binary_features(df)
        
        if create_lags:
            df = self.create_lag_features(df)
            df = self.create_rolling_features(df)
            
        # Drop rows with NaN values created by lag features
        df = df.dropna()
        
        return df

    def get_feature_names(self):
        """Return list of processed feature names."""
        return {
            'categorical': [f'{col}_Encoded' for col in self.encoders.keys()],
            'numerical': [f'{col}_Scaled' for col in self.scalers.keys()],
            'temporal': ['DayOfWeek', 'Month', 'Year', 'DayOfMonth', 'WeekOfYear'],
            'binary': ['HasPromotion']
        }