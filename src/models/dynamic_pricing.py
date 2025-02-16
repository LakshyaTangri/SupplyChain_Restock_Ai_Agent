# src/models/dynamic_pricing.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class DynamicPricingOptimizer:
    def __init__(self, min_margin=0.2, max_price_change=0.15):
        """
        Initialize Dynamic Pricing Optimizer
        
        Parameters:
        - min_margin: Minimum profit margin to maintain
        - max_price_change: Maximum allowed price change (as percentage)
        """
        self.price_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        self.demand_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        self.min_margin = min_margin
        self.max_price_change = max_price_change
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for price optimization."""
        df = df.copy()
        
        # Time-based features
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Competition features
        df['Price_Ratio'] = df['Price'] / df['Competitor Pricing']
        
        # Historical sales features
        for lag in [1, 7]:
            df[f'Sales_Lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(lag)
            df[f'Price_Lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])['Price'].shift(lag)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train(self, df):
        """Train price optimization models."""
        prepared_data = self.prepare_features(df)
        
        # Prepare feature sets
        price_features = [
            'DayOfWeek', 'Month', 'IsWeekend', 'Competitor Pricing',
            'Inventory Level', 'Seasonality'
        ]
        
        demand_features = price_features + ['Price', 'Discount']
        
        # Scale features
        X_price = self.scaler.fit_transform(prepared_data[price_features])
        X_demand = prepared_data[demand_features]
        
        # Train price prediction model
        self.price_model.fit(X_price, prepared_data['Price'])
        
        # Train demand prediction model
        self.demand_model.fit(X_demand, prepared_data['Units Sold'])
        
        return self
    
    def optimize_price(self, current_data, cost_data):
        """
        Optimize prices based on current market conditions and costs.
        
        Parameters:
        - current_data: DataFrame with current market conditions
        - cost_data: DataFrame with product costs
        
        Returns:
        DataFrame with optimized prices
        """
        prepared_data = self.prepare_features(current_data)
        
        results = []
        price_range = np.linspace(0.8, 1.2, 50)  # Test price adjustments ±20%
        
        for _, row in prepared_data.iterrows():
            current_price = row['Price']
            product_cost = cost_data[
                cost_data['Product ID'] == row['Product ID']
            ]['Cost'].iloc[0]
            
            best_price = current_price
            best_revenue = 0
            
            # Test different price points
            for price_adj in price_range:
                test_price = current_price * price_adj
                
                # Skip if price doesn't meet margin requirements
                if (test_price - product_cost) / test_price < self.min_margin:
                    continue
                    
                # Skip if price change is too large
                if abs(test_price - current_price) / current_price > self.max_price_change:
                    continue
                
                # Predict demand at this price
                row_copy = row.copy()
                row_copy['Price'] = test_price
                predicted_demand = self.demand_model.predict([row_copy])[0]
                
                # Calculate revenue
                revenue = predicted_demand * (test_price - product_cost)
                
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_price = test_price
            
            results.append({
                'Store ID': row['Store ID'],
                'Product ID': row['Product ID'],
                'Current Price': current_price,
                'Optimized Price': round(best_price, 2),
                'Price Change': round((best_price - current_price) / current_price * 100, 1),
                'Expected Revenue': round(best_revenue, 2)
            })
        
        return pd.DataFrame(results)
    
    def generate_pricing_report(self, current_data, cost_data, competitor_data):
        """Generate comprehensive pricing report."""
        # Get optimized prices
        optimized_prices = self.optimize_price(current_data, cost_data)
        
        # Add competitor analysis
        report = optimized_prices.merge(
            competitor_data[['Product ID', 'Competitor Pricing']],
            on='Product ID'
        )
        
        # Add price positioning metrics
        report['Price vs Competitor'] = round(
            (report['Optimized Price'] - report['Competitor Pricing']) / 
            report['Competitor Pricing'] * 100, 1
        )
        
        # Add margin analysis
        report = report.merge(
            cost_data[['Product ID', 'Cost']],
            on='Product ID'
        )
        report['Current Margin'] = round(
            (report['Current Price'] - report['Cost']) / 
            report['Current Price'] * 100, 1
        )
        report['Optimized Margin'] = round(
            (report['Optimized Price'] - report['Cost']) / 
            report['Optimized Price'] * 100, 1
        )
        
        return report
    
    def save_models(self, filepath_prefix):
        """Save trained models to disk."""
        joblib.dump(self.price_model, f'{filepath_prefix}_price_model.pkl')
        joblib.dump(self.demand_model, f'{filepath_prefix}_demand_model.pkl')
        joblib.dump(self.scaler, f'{filepath_prefix}_scaler.pkl')
    
    def load_models(self, filepath_prefix):
        """Load trained models from disk."""
        self.price_model = joblib.load(f'{filepath_prefix}_price_model.pkl')
        self.demand_model = joblib.load(f'{filepath_prefix}_demand_model.pkl')
        self.scaler = joblib.load(f'{filepath_prefix}_scaler.pkl')
        return self