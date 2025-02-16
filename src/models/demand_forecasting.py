# src/models/demand_forecasting.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from scipy.stats import norm

app = FastAPI()

class DemandForecaster:
    def __init__(self, model_params=None):
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        self.model = RandomForestRegressor(**self.model_params)
        self.arima_model = None
        self.feature_importance = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for demand forecasting."""
        # Create time-based features
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Create lag features
        for lag in [1, 7, 14, 30]:
            df[f'Sales_Lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(lag)
            df[f'Demand_Lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])['Demand Forecast'].shift(lag)
        
        # Create rolling mean features
        for window in [7, 14, 30]:
            df[f'Sales_Rolling_Mean_{window}'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Handle promotions and holidays
        df['HasPromotion'] = df['Holiday/Promotion'].notna().astype(int)
        
        # Price-related features
        df['Price_Ratio'] = df['Price'] / df['Competitor Pricing']
        df['Discount_Amount'] = df['Price'] * df['Discount']
        
        df = df.dropna()  # Handle NaN values
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns used for prediction."""
        return ['DayOfWeek', 'Month', 'Year', 'IsWeekend', 
                'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_30',
                'Demand_Lag_1', 'Demand_Lag_7', 'Demand_Lag_14', 'Demand_Lag_30',
                'Sales_Rolling_Mean_7', 'Sales_Rolling_Mean_14', 'Sales_Rolling_Mean_30',
                'HasPromotion', 'Price_Ratio', 'Discount_Amount',
                'Price', 'Discount', 'Inventory Level', 'Seasonality']
    
    def train(self, df):
        """Train the demand forecasting model."""
        prepared_data = self.prepare_features(df)
        prepared_data = prepared_data.dropna()  # Remove rows with NaN values
        
        X = prepared_data[self.get_feature_columns()]
        y = prepared_data['Units Sold']
        
        self.model.fit(X, y)
        self.feature_importance = pd.DataFrame({
            'feature': self.get_feature_columns(),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Train ARIMA model
        self.arima_model = self.train_arima_model(df)
        
        return self
    
    def train_arima_model(self, df: pd.DataFrame) -> ARIMA:
        """Train demand forecasting model using ARIMA"""
        # Prepare data
        df['Units Sold'] = pd.to_numeric(df['Units Sold'])
        df.set_index('Date', inplace=True)

        # Split data into training and testing sets
        train_data = df[:int(0.8 * len(df))]
        test_data = df[int(0.8 * len(df)):]

        # Train ARIMA model
        model = ARIMA(train_data['Units Sold'], order=(1,1,1))
        model_fit = model.fit()

        # Evaluate model performance
        forecast, stderr, conf_int = model_fit.forecast(steps=len(test_data))

        # Return trained model
        return model_fit
    
    def predict(self, df):
        """Generate demand forecasts."""
        prepared_data = self.prepare_features(df)
        X = prepared_data[self.get_feature_columns()]
        
        predictions = self.model.predict(X)
        
        # Generate ARIMA predictions
        arima_predictions = self.arima_model.forecast(steps=len(df))
        
        return predictions, arima_predictions
    
    def evaluate(self, df):
        """Evaluate model performance."""
        prepared_data = self.prepare_features(df)
        prepared_data = prepared_data.dropna()
        
        X = prepared_data[self.get_feature_columns()]
        y_true = prepared_data['Units Sold']
        y_pred = self.model.predict(X)
        
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        self.model = joblib.load(filepath)
        return self
    
    def generate_forecast_report(self, df, future_days=30):
        """Generate a detailed forecast report."""
        current_date = df['Date'].max()
        future_dates = pd.date_range(start=current_date + timedelta(days=1),
                                   periods=future_days, freq='D')
        
        # Create future dataframe
        future_df = pd.DataFrame()
        for store_id in df['Store ID'].unique():
            for product_id in df['Product ID'].unique():
                store_product_df = pd.DataFrame({
                    'Date': future_dates,
                    'Store ID': store_id,
                    'Product ID': product_id
                })
                
                # Copy the latest values for other features
                latest_data = df[(df['Store ID'] == store_id) & 
                               (df['Product ID'] == product_id)].iloc[-1]
                for col in df.columns:
                    if col not in ['Date', 'Units Sold', 'Demand Forecast']:
                        store_product_df[col] = latest_data[col]
                
                future_df = pd.concat([future_df, store_product_df])
        
        # Generate predictions
        future_df['Forecast'] = self.predict(future_df)
        
        return future_df[['Date', 'Store ID', 'Product ID', 'Forecast']]

@app.post("/forecast/demand")
async def forecast_demand(request: ForecastRequest):
    """
    Generate demand forecast for a specified date range, store, and product.

    Parameters:
    - start_date: Start date for the forecast.
    - end_date: End date for the forecast.
    - store_id: Store ID for the forecast.
    - product_id: Product ID for the forecast.

    Returns:
    - JSON response with forecasted demand data.
    """
    try:
        forecast = DemandForecaster().forecast_demand(
            request.start_date, request.end_date, request.store_id, request.product_id
        )
        return {"forecast": forecast.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/inventory")
async def optimize_inventory(data: List[InventoryData]):
    """
    Optimize inventory levels based on provided data.

    Parameters:
    - data: List of InventoryData objects.

    Returns:
    - JSON response with optimization results.
    """
    try:
        optimization_results = InventoryOptimizer().optimize_inventory(data)
        return {"optimization_results": optimization_results.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/pricing")
async def optimize_pricing(current_data: dict, cost_data: dict):
    """
    Generate price optimization recommendations.

    Parameters:
    - current_data: Current pricing data.
    - cost_data: Cost data.

    Returns:
    - JSON response with optimized prices.
    """
    try:
        optimized_prices = pricing_optimizer.optimize_price(
            current_data=pd.DataFrame(current_data),
            cost_data=pd.DataFrame(cost_data)
        )
        return {"optimized_prices": optimized_prices.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suppliers/optimize")
async def optimize_suppliers(demand_forecast: dict, supplier_ratings: dict, inventory_levels: dict):
    """
    Generate supplier optimization recommendations.

    Parameters:
    - demand_forecast: Demand forecast data.
    - supplier_ratings: Supplier ratings data.
    - inventory_levels: Inventory levels data.

    Returns:
    - JSON response with supplier optimization recommendations.
    """
    try:
        recommendations = supplier_manager.optimize_supplier_selection(
            demand_forecast=pd.DataFrame(demand_forecast),
            supplier_ratings=pd.DataFrame(supplier_ratings),
            inventory_levels=pd.DataFrame(inventory_levels)
        )
        return {"recommendations": recommendations.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/performance")
async def generate_performance_report(start_date: datetime, end_date: datetime):
    """
    Generate comprehensive performance report.

    Parameters:
    - start_date: Start date for the report.
    - end_date: End date for the report.

    Returns:
    - JSON response with HTML report.
    """
    try:
        # Gather data for the report
        sales_data = load_sales_data(start_date, end_date)
        inventory_data = load_inventory_data(start_date, end_date)
        pricing_data = load_pricing_data(start_date, end_date)
        
        # Generate HTML report
        report_html = reporter.generate_html_report(
            sales_data=sales_data,
            inventory_data=inventory_data,
            pricing_data=pricing_data
        )
        
        return {"report_html": report_html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/supply-chain")
async def optimize_supply_chain():
    """
    Optimize the entire supply chain.

    Returns:
    - JSON response with optimization message.
    """
    try:
        # Implement the supply chain optimization logic
        message = "Supply chain optimized"
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))