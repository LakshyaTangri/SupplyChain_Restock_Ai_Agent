# src/models/supplier_management.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()

class SupplierManager:
    def __init__(self):
        self.supplier_ratings = {}
        self.order_history = pd.DataFrame()
        self.scaler = MinMaxScaler()
        
    def evaluate_supplier_performance(self, supplier_data):
        """
        Evaluate supplier performance based on multiple criteria.
        """
        evaluations = []
        
        for supplier_id, data in supplier_data.groupby('Supplier ID'):
            # Calculate key metrics
            on_time_delivery = (data['Actual Delivery Date'] <= data['Expected Delivery Date']).mean()
            quality_rating = data['Quality Rating'].mean()
            price_competitiveness = 1 - (data['Price'] / data['Market Price']).mean()
            lead_time = (data['Actual Delivery Date'] - data['Order Date']).dt.days.mean()
            
            # Calculate reliability score
            order_completion_rate = (data['Delivered Quantity'] / data['Ordered Quantity']).mean()
            
            # Composite score calculation
            score = (on_time_delivery * 0.3 +
                    quality_rating * 0.25 +
                    price_competitiveness * 0.2 +
                    (1 - lead_time/30) * 0.15 +  # Normalize lead time to 30 days
                    order_completion_rate * 0.1)
            
            evaluations.append({
                'Supplier ID': supplier_id,
                'On-Time Delivery Rate': round(on_time_delivery * 100, 2),
                'Quality Rating': round(quality_rating, 2),
                'Price Competitiveness': round(price_competitiveness * 100, 2),
                'Average Lead Time': round(lead_time, 1),
                'Order Completion Rate': round(order_completion_rate * 100, 2),
                'Performance Score': round(score * 100, 2)
            })
        
        return pd.DataFrame(evaluations)
    
    def optimize_supplier_selection(self, demand_forecast, supplier_ratings, inventory_levels):
        """
        Optimize supplier selection based on demand forecast and supplier ratings.
        """
        recommendations = []
        
        for product_id, demand in demand_forecast.groupby('Product ID'):
            total_demand = demand['Forecast'].sum()
            current_inventory = inventory_levels[
                inventory_levels['Product ID'] == product_id
            ]['Inventory Level'].sum()
            
            required_quantity = max(0, total_demand - current_inventory)
            
            if required_quantity > 0:
                # Get suppliers for this product
                product_suppliers = supplier_ratings[
                    supplier_ratings['Product ID'] == product_id
                ].copy()
                
                # Calculate optimal order allocation
                product_suppliers['Order Quantity'] = self._allocate_orders(
                    required_quantity,
                    product_suppliers
                )
                
                recommendations.extend(product_suppliers[
                    product_suppliers['Order Quantity'] > 0
                ].to_dict('records'))
        
        return pd.DataFrame(recommendations)
    
    def _allocate_orders(self, total_quantity, suppliers):
        """
        Allocate order quantities among suppliers based on their ratings.
        """
        # Normalize performance scores
        normalized_scores = suppliers['Performance Score'] / suppliers['Performance Score'].sum()
        
        # Initial allocation based on performance
        base_allocation = normalized_scores * total_quantity
        
        # Adjust for minimum order quantities and capacity constraints
        adjusted_allocation = np.minimum(
            base_allocation,
            suppliers['Maximum Capacity']
        )
        
        # Ensure minimum order quantities are met or set to 0
        adjusted_allocation = np.where(
            adjusted_allocation < suppliers['Minimum Order Quantity'],
            0,
            adjusted_allocation
        )
        
        return np.round(adjusted_allocation)
    
    def generate_purchase_orders(self, supplier_recommendations, lead_times):
        """
        Generate purchase orders based on supplier recommendations.
        """
        purchase_orders = []
        
        for _, recommendation in supplier_recommendations.iterrows():
            expected_delivery = datetime.now() + timedelta(
                days=lead_times[recommendation['Supplier ID']]
            )
            
            po = {
                'PO Number': f"PO-{datetime.now().strftime('%Y%m%d')}-{len(purchase_orders):04d}",
                'Supplier ID': recommendation['Supplier ID'],
                'Product ID': recommendation['Product ID'],
                'Order Quantity': recommendation['Order Quantity'],
                'Order Date': datetime.now(),
                'Expected Delivery': expected_delivery,
                'Status': 'Created'
            }
            
            purchase_orders.append(po)
        
        return pd.DataFrame(purchase_orders)
    
    def track_order_status(self, purchase_orders, delivery_updates):
        """
        Track and update purchase order status.
        """
        tracked_orders = purchase_orders.copy()
        
        for _, update in delivery_updates.iterrows():
            mask = tracked_orders['PO Number'] == update['PO Number']
            tracked_orders.loc[mask, 'Status'] = update['Status']
            
            if update['Status'] == 'Delivered':
                tracked_orders.loc[mask, 'Actual Delivery'] = update['Delivery Date']
                tracked_orders.loc[mask, 'Delivery Delay'] = (
                    update['Delivery Date'] - tracked_orders.loc[mask, 'Expected Delivery']
                ).dt.days
        
        return tracked_orders
    
    def generate_supplier_report(self, supplier_performance, purchase_orders):
        """
        Generate comprehensive supplier performance report.
        """
        report = supplier_performance.copy()
        
        # Add order history metrics
        order_metrics = purchase_orders.groupby('Supplier ID').agg({
            'Order Quantity': ['sum', 'count'],
            'Delivery Delay': 'mean'
        }).round(2)
        
        report = report.merge(
            order_metrics,
            left_on='Supplier ID',
            right_index=True,
            how='left'
        )
        
        # Calculate risk scores
        report['Risk Score'] = self._calculate_risk_score(report)
        
        # Add recommendations
        report['Recommendation'] = np.where(
            report['Risk Score'] > 0.7,
            'Reduce Orders',
            np.where(
                report['Risk Score'] < 0.3,
                'Increase Orders',
                'Maintain Current Level'
            )
        )
        
        return report
    
    def _calculate_risk_score(self, supplier_data):
        """
        Calculate risk score for each supplier.
        """
        risk_factors = [
            'Delivery Delay',
            'Order Completion Rate',
            'Quality Rating'
        ]
        
        # Normalize risk factors
        normalized_data = self.scaler.fit_transform(supplier_data[risk_factors])
        
        # Calculate weighted risk score
        weights = [0.4, 0.3, 0.3]  # Weights for each factor
        risk_scores = np.dot(normalized_data, weights)
        
        return risk_scores.round(2)
    
    def manage_suppliers(self, data: pd.DataFrame, optimal_inventory: pd.DataFrame) -> pd.DataFrame:
        """Manage suppliers using supplier scoring model"""
        # Calculate supplier scores based on performance
        supplier_scores = data.groupby('Supplier ID')['Units Sold'].sum() / optimal_inventory['Optimal Inventory'].sum()

        # Create a DataFrame with supplier scores
        supplier_scores_df = pd.DataFrame({'Supplier ID': supplier_scores.index, 'Score': supplier_scores.values})

        # Negotiate prices with suppliers based on scores
        negotiated_prices = supplier_scores_df.apply(lambda x: x['Score'] * 0.9 if x['Score'] > 0.8 else x['Score'] * 0.95, axis=1)

        # Create a DataFrame with negotiated prices
        negotiated_prices_df = pd.DataFrame({'Supplier ID': supplier_scores_df['Supplier ID'], 'Negotiated Price': negotiated_prices.values})

        return negotiated_prices_df

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
        recommendations = SupplierManager().optimize_supplier_selection(
            demand_forecast=pd.DataFrame(demand_forecast),
            supplier_ratings=pd.DataFrame(supplier_ratings),
            inventory_levels=pd.DataFrame(inventory_levels)
        )
        return {"recommendations": recommendations.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))