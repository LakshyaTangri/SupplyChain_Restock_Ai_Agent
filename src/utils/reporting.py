import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class SupplyChainReporter:
    def __init__(self):
        # Check if 'seaborn' is available, otherwise use a default style
        if 'seaborn' in plt.style.available:
            plt.style.use('seaborn')
        else:
            plt.style.use('ggplot')  # Use an alternative style if 'seaborn' is not available
        # ...existing code...
        
    def generate_executive_summary(self, sales_data, inventory_data, pricing_data):
        """Generate executive summary with key metrics."""
        summary = {
            'Total Revenue': sales_data['Units Sold'] * sales_data['Price'],
            'Total Units Sold': sales_data['Units Sold'].sum(),
            'Average Order Value': (sales_data['Units Sold'] * sales_data['Price']).mean(),
            'Stock Turnover Rate': sales_data['Units Sold'].sum() / inventory_data['Inventory Level'].mean(),
            'Average Margin': ((sales_data['Price'] - pricing_data['Cost']) / sales_data['Price']).mean() * 100
        }
        
        # Add week-over-week growth
        latest_week = sales_data.groupby(pd.Grouper(key='Date', freq='W'))['Units Sold'].sum().iloc[-1]
        previous_week = sales_data.groupby(pd.Grouper(key='Date', freq='W'))['Units Sold'].sum().iloc[-2]
        summary['Weekly Growth'] = (latest_week - previous_week) / previous_week * 100
        
        return pd.Series(summary)
    
    def generate_sales_report(self, sales_data, grouping='daily'):
        """Generate sales analysis report."""
        if grouping == 'daily':
            freq = 'D'
        elif grouping == 'weekly':
            freq = 'W'
        elif grouping == 'monthly':
            freq = 'M'
        
        # Sales trends
        sales_trends = sales_data.groupby(
            [pd.Grouper(key='Date', freq=freq), 'Category']
        ).agg({
            'Units Sold': 'sum',
            'Price': 'mean',
            'Revenue': lambda x: (x['Units Sold'] * x['Price']).sum()
        }).reset_index()
        
        # Top performing products
        top_products = sales_data.groupby('Product ID').agg({
            'Units Sold': 'sum',
            'Revenue': lambda x: (x['Units Sold'] * x['Price']).sum()
        }).sort_values('Revenue', ascending=False).head(10)
        
        # Regional performance
        regional_performance = sales_data.groupby('Region').agg({
            'Units Sold': 'sum',
            'Revenue': lambda x: (x['Units Sold'] * x['Price']).sum()
        })
        
        return {
            'sales_trends': sales_trends,
            'top_products': top_products,
            'regional_performance': regional_performance
        }
    
    def generate_inventory_report(self, inventory_data, sales_data):
        """Generate inventory analysis report."""
        # Current inventory status
        current_inventory = inventory_data.groupby(['Store ID', 'Product ID']).agg({
            'Inventory Level': 'last',
            'Units Ordered': 'sum'
        })
        
        # Days of inventory
        daily_sales = sales_data.groupby(['Store ID', 'Product ID'])['Units Sold'].mean()
        current_inventory['Days of Inventory'] = current_inventory['Inventory Level'] / daily_sales
        
        # Stockout risk
        current_inventory['Stockout Risk'] = np.where(
            current_inventory['Days of Inventory'] < 7, 'High',
            np.where(current_inventory['Days of Inventory'] < 14, 'Medium', 'Low')
        )
        
        return current_inventory
    
    def plot_sales_trends(self, sales_data, save_path=None):
        """Plot sales trends visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Daily sales trend
        daily_sales = sales_data.groupby('Date')['Units Sold'].sum()
        ax1.plot(daily_sales.index, daily_sales.values)
        ax1.set_title('Daily Sales Trend')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Units Sold')
        
        # Category distribution
        category_sales = sales_data.groupby('Category')['Units Sold'].sum()
        category_sales.plot(kind='bar', ax=ax2)
        ax2.set_title('Sales by Category')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Units Sold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_inventory_heatmap(self, inventory_data, save_path=None):
        """Plot inventory heatmap by store and product."""
        pivot_data = inventory_data.pivot(
            index='Store ID',
            columns='Product ID',
            values='Inventory Level'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='YlOrRd')
        plt.title('Inventory Levels by Store and Product')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def generate_html_report(self, sales_data, inventory_data, pricing_data):
        """Generate HTML report with all analyses."""
        exec_summary = self.generate_executive_summary(sales_data, inventory_data, pricing_data)
        sales_report = self.generate_sales_report(sales_data)
        inventory_report = self.generate_inventory_report(inventory_data, sales_data)
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .metric {{ margin: 10px; padding: 10px; border: 1px solid #ddd; }}
                .warning {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Supply Chain Performance Report</h1>
            <h2>Executive Summary</h2>
            <div class="metric">
                <h3>Key Metrics</h3>
                <p>Total Revenue: ${exec_summary['Total Revenue']:,.2f}</p>
                <p>Total Units Sold: {exec_summary['Total Units Sold']:,}</p>
                <p>Average Order Value: ${exec_summary['Average Order Value']:.2f}</p>
                <p>Stock Turnover Rate: {exec_summary['Stock Turnover Rate']:.2f}</p>
                <p>Average Margin: {exec_summary['Average Margin']:.1f}%</p>
                <p>Weekly Growth: {exec_summary['Weekly Growth']:.1f}%</p>
            </div>
            
            <h2>Inventory Status</h2>
            <div class="metric">
                <h3>Stockout Risks</h3>
                {inventory_report[inventory_report['Stockout Risk'] == 'High'].to_html()}
            </div>
            
            <h2>Top Performing Products</h2>
            <div class="metric">
                {sales_report['top_products'].to_html()}
            </div>
            
            <h2>Regional Performance</h2>
            <div class="metric">
                {sales_report['regional_performance'].to_html()}
            </div>
        </body>
        </html>
        """
        
        return html