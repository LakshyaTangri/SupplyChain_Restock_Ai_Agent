import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import logging

class DataPreprocessor:
    """
    Handles data preprocessing including loading, cleaning, and validation.
    """
    
    def __init__(self, schema: SupplyChainSchema):
        self.schema = schema
        self.encoders = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe by handling missing values and outliers."""
        try:
            # Handle missing values
            df['inventory_level'].fillna(df['inventory_level'].median(), inplace=True)
            df['units_sold'].fillna(0, inplace=True)
            df['units_ordered'].fillna(0, inplace=True)
            df['demand_forecast'].fillna(df['demand_forecast'].mean(), inplace=True)
            
            # Remove outliers using IQR method
            for col in self.schema.NUMERICAL_COLUMNS:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            
            self.logger.info("Data cleaning completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate that the dataframe matches the expected schema."""
        try:
            # Check columns
            if not all(col in df.columns for col in self.schema.COLUMNS):
                missing_cols = set(self.schema.COLUMNS) - set(df.columns)
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Check data types
            for col, dtype in self.schema.SCHEMA_RULES.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            
            self.logger.info("Schema validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            raise
    
    def encode_categorical(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        try:
            df_encoded = df.copy()
            
            for col in self.schema.CATEGORICAL_COLUMNS:
                if training:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df[col])
                else:
                    df_encoded[col] = self.encoders[col].transform(df[col])
            
            self.logger.info("Categorical encoding completed")
            return df_encoded
            
        except Exception as e:
            self.logger.error(f"Error encoding categorical variables: {str(e)}")
            raise
    
    def normalize_numerical(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """Normalize numerical variables."""
        try:
            df_normalized = df.copy()
            
            if training:
                df_normalized[self.schema.NUMERICAL_COLUMNS] = self.scaler.fit_transform(
                    df[self.schema.NUMERICAL_COLUMNS]
                )
            else:
                df_normalized[self.schema.NUMERICAL_COLUMNS] = self.scaler.transform(
                    df[self.schema.NUMERICAL_COLUMNS]
                )
            
            self.logger.info("Numerical normalization completed")
            return df_normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing numerical variables: {str(e)}")
            raise

class FeatureEngineer:
    """
    Handles feature engineering for the supply chain data.
    """
    
    def __init__(self, schema: SupplyChainSchema):
        self.schema = schema
        self.logger = logging.getLogger(__name__)
    
    def calculate_moving_averages(self, df: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Calculate moving averages for numerical columns."""
        try:
            df_features = df.copy()
            
            for col in ['units_sold', 'demand_forecast']:
                for window in windows:
                    df_features[f'{col}_ma_{window}'] = df_features.groupby('product_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
            
            self.logger.info("Moving averages calculated successfully")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {str(e)}")
            raise
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 7, 14]) -> pd.DataFrame:
        """Create lag features for time series analysis."""
        try:
            df_features = df.copy()
            
            for col in ['units_sold', 'demand_forecast', 'price']:
                for lag in lags:
                    df_features[f'{col}_lag_{lag}'] = df_features.groupby('product_id')[col].shift(lag)
            
            self.logger.info("Lag features created successfully")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-related features."""
        try:
            df_features = df.copy()
            
            # Price difference from competitor
            df_features['price_diff'] = df_features['price'] - df_features['competitor_pricing']
            
            # Price ratio
            df_features['price_ratio'] = df_features['price'] / df_features['competitor_pricing']
            
            # Effective price after discount
            df_features['effective_price'] = df_features['price'] * (1 - df_features['discount']/100)
            
            self.logger.info("Price features created successfully")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating price features: {str(e)}")
            raise
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal features from date."""
        try:
            df_features = df.copy()
            df_features['date'] = pd.to_datetime(df_features['date'])
            
            # Extract date components
            df_features['day_of_week'] = df_features['date'].dt.dayofweek
            df_features['month'] = df_features['date'].dt.month
            df_features['quarter'] = df_features['date'].dt.quarter
            df_features['year'] = df_features['date'].dt.year
            
            # Create cyclical features
            df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week']/7)
            df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week']/7)
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month']/12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month']/12)
            
            self.logger.info("Seasonal features created successfully")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating seasonal features: {str(e)}")
            raise