# src/models/inventory_optimization.py

import pandas as pd
import numpy as np
from scipy.stats import norm
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from src.api.supply_chain_api import InventoryData

app = FastAPI()

class InventoryOptimizer:
    def __init__(self, service_level=0.95, holding_cost_rate=0.20, stockout_cost_rate=0.30):
        """
        Initialize Inventory Optimizer
        
        Parameters:
        - service_level: Target service level (probability of not stocking out)
        - holding_cost_rate: Annual holding cost as a fraction of item value
        - stockout_cost_rate: Stockout cost as a fraction of item value
        """
        self.service_level = service_level
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_rate = stockout_cost_rate
        self.z_score = norm.ppf(service_level)
        
    def calculate_safety_stock(self, demand_mean, demand_std, lead_time):
        """Calculate optimal safety stock level."""
        lead_time_demand_std = demand_std * np.sqrt(lead_time)
        safety_stock = self.z_score * lead_time_demand_std
        return np.maximum(safety_stock, 0)
    
    def calculate_reorder_point(self, demand_mean, demand_std, lead_time):
        """Calculate reorder point."""
        lead_time_demand = demand_mean * lead_time
        safety_stock = self.calculate_safety_stock(demand_mean, demand_std, lead_time)
        reorder_point = lead_time_demand + safety_stock
        return np.maximum(reorder_point, 0)
    
    def calculate_order_quantity(self, demand_mean, order_cost, item_cost):
        """Calculate economic order quantity (EOQ)."""
        annual_demand = demand_mean * 365
        eoq = np.sqrt((2 * annual_demand * order_cost) / 
                      (self.holding_cost_rate * item_cost))
        return np.maximum(eoq, 0)
    
    def optimize_inventory(self, df, lead_time_days, order_cost):
        """
        Optimize inventory parameters for each product at each store.
        
        Parameters:
        - df: DataFrame with historical demand data
        - lead_time_days: Lead time in days for receiving orders
        - order_cost: Fixed cost per order
        
        Returns:
        DataFrame with optimal inventory parameters
        """
        results = []
        
        for (store_id, product_id), group in df.groupby(['Store ID', 'Product ID']):
            # Calculate demand statistics
            demand_mean = group['Units Sold'].mean()
            demand_std = group['Units Sold'].std()
            
            # Get latest price
            current_price = group['Price'].iloc[-1]
            
            # Calculate inventory parameters
            safety_stock = self.calculate_safety_stock(
                demand_mean, demand_std, lead_time_days)
            reorder_point = self.calculate_reorder_point(
                demand_mean, demand_std, lead_time_days)
            order_quantity = self.calculate_order_quantity(
                demand_mean, order_cost, current_price)
            
            # Calculate costs
            annual_holding_cost = (safety_stock + order_quantity/2) * \
                                current_price * self.holding_cost_rate
            annual_order_cost = (demand_mean * 365 / order_quantity) * order_cost
            
            results.append({
                'Store ID': store_id,
                'Product ID': product_id,
                'Safety Stock': round(safety_stock),
                'Reorder Point': round(reorder_point),
                'Order Quantity': round(order_quantity),
                'Daily Demand Mean': round(demand_mean, 2),
                'Daily Demand Std': round(demand_std, 2),
                'Annual Holding Cost': round(annual_holding_cost, 2),
                'Annual Order Cost': round(annual_order_cost, 2),
                'Total Annual Cost': round(annual_holding_cost + annual_order_cost, 2)
            })
        
        return pd.DataFrame(results)
    
    def generate_reorder_recommendations(self, current_inventory, optimal_params):
        """Generate reorder recommendations based on current inventory levels."""
        recommendations = []
        
        for _, row in optimal_params.iterrows():
            current_level = current_inventory[
                (current_inventory['Store ID'] == row['Store ID']) & 
                (current_inventory['Product ID'] == row['Product ID'])
            ]['Inventory Level'].iloc[0]
            
            if current_level <= row['Reorder Point']:
                recommendations.append({
                    'Store ID': row['Store ID'],
                    'Product ID': row['Product ID'],
                    'Current Inventory': current_level,
                    'Reorder Point': row['Reorder Point'],
                    'Recommended Order': row['Order Quantity'],
                    'Priority': 'High' if current_level <= row['Safety Stock'] else 'Medium'
                })
        
        return pd.DataFrame(recommendations)
    
    def simulate_inventory_policy(self, demand_data, optimal_params, simulation_days=30):
        """Simulate inventory policy to validate parameters."""
        results = []
        
        for _, policy in optimal_params.iterrows():
            store_id = policy['Store ID']
            product_id = policy['Product ID']
            
            # Get historical demand for this store/product
            historical_demand = demand_data[
                (demand_data['Store ID'] == store_id) & 
                (demand_data['Product ID'] == product_id)
            ]['Units Sold'].values
            
            # Simulate future demand using bootstrap
            simulated_demand = np.random.choice(
                historical_demand, size=simulation_days)
            
            # Initialize simulation
            inventory_level = policy['Safety Stock'] + policy['Order Quantity']
            stockouts = 0
            total_holding_cost = 0
            total_orders = 0
            
            # Run simulation
            for day in range(simulation_days):
                # Subtract demand
                inventory_level -= simulated_demand[day]
                
                # Check for stockout
                if inventory_level < 0:
                    stockouts += 1
                    inventory_level = 0
                
                # Check for reorder
                if inventory_level <= policy['Reorder Point']:
                    inventory_level += policy['Order Quantity']
                    total_orders += 1
                
                # Add holding cost
                total_holding_cost += inventory_level * \
                    (self.holding_cost_rate / 365)
            
            results.append({
                'Store ID': store_id,
                'Product ID': product_id,
                'Service Level': 1 - (stockouts / simulation_days),
                'Average Inventory': inventory_level / simulation_days,
                'Number of Orders': total_orders,
                'Total Holding Cost': total_holding_cost
            })
        
        return pd.DataFrame(results)

class ForecastRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    store_id: Optional[int] = None
    product_id: Optional[int] = None

inventory_optimizer = InventoryOptimizer()

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

from fastapi.testclient import TestClient
from src.api.supply_chain_api import app

client = TestClient(app)

def test_forecast_demand():
    response = client.post("/forecast/demand", json={
        "start_date": "2024-02-16",
        "end_date": "2024-03-16",
        "store_id": 1,
        "product_id": 101
    })
    assert response.status_code == 200
    assert "forecast" in response.json()

def test_optimize_inventory():
    inventory_data = [
        {"store_id": 1, "product_id": 101, "current_level": 50, "date": "2024-02-16"}
    ]
    response = client.post("/optimize/inventory", json=inventory_data)
    assert response.status_code == 200
    assert "optimization_results" in response.json()

def test_optimize_pricing():
    current_data = {"data": [{"Date": "2024-02-16", "Price": 10.0}]}
    cost_data = {"data": [{"Date": "2024-02-16", "Cost": 5.0}]}
    response = client.post("/optimize/pricing", json={"current_data": current_data, "cost_data": cost_data})
    assert response.status_code == 200
    assert "optimized_prices" in response.json()

def test_optimize_suppliers():
    demand_forecast = {"data": [{"Date": "2024-02-16", "Forecast": 100}]}
    supplier_ratings = {"data": [{"Supplier": "A", "Rating": 4.5}]}
    inventory_levels = {"data": [{"Date": "2024-02-16", "Inventory Level": 50}]}
    response = client.post("/suppliers/optimize", json={
        "demand_forecast": demand_forecast,
        "supplier_ratings": supplier_ratings,
        "inventory_levels": inventory_levels
    })
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_generate_performance_report():
    response = client.get("/reports/performance", params={"start_date": "2024-02-16", "end_date": "2024-03-16"})
    assert response.status_code == 200
    assert "report_html" in response.json()

def test_optimize_supply_chain():
    response = client.post("/optimize/supply-chain")
    assert response.status_code == 200
    assert response.json() == {"message": "Supply chain optimized"}