import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd
import logging

class MetricsVisualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def plot_demand_forecast(self, historical_data: pd.DataFrame, forecast_data: pd.DataFrame):
        """Plot historical demand and forecasts."""
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['units_sold'],
            name='Historical Demand',
            line=dict(color='blue')
        ))
        
        # Add forecast data
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['forecast'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Demand Forecast',
            xaxis_title='Date',
            yaxis_title='Units',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
    
    def plot_inventory_levels(self, inventory_data: pd.DataFrame):
        """Plot current inventory levels and reorder points."""
        fig = go.Figure()
        
        # Add inventory levels
        fig.add_trace(go.Bar(
            x=inventory_data['product_id'],
            y=inventory_data['current_level'],
            name='Current Inventory'
        ))
        
        # Add reorder points
        fig.add_trace(go.Scatter(
            x=inventory_data['product_id'],
            y=inventory_data['reorder_point'],
            name='Reorder Point',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Inventory Levels by Product',
            xaxis_title='Product ID',
            yaxis_title='Units',
            barmode='group'
        )
        
        st.plotly_chart(fig)
    
    def plot_price_optimization(self, price_data: pd.DataFrame):
        """Plot price optimization results."""
        fig = go.Figure()
        
        # Add current prices
        fig.add_trace(go.Bar(
            x=price_data['product_id'],
            y=price_data['current_price'],
            name='Current Price'
        ))
        
        # Add optimized prices
        fig.add_trace(go.Bar(
            x=price_data['product_id'],
            y=price_data['optimized_price'],
            name='Optimized Price'
        ))
        
        fig.update_layout(
            title='Price Optimization Results',
            xaxis_title='Product ID',
            yaxis_title='Price ($)',
            barmode='group'
        )
        
        st.plotly_chart(fig)