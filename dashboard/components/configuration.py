import streamlit as st
from typing import Dict
import json
import logging

class ModelConfiguration:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.default_config = {
            'demand_forecast': {
                'horizon': 30,
                'confidence_level': 0.95,
                'update_frequency': 'daily'
            },
            'inventory': {
                'safety_stock_level': 0.95,
                'lead_time_days': 7,
                'reorder_point_multiplier': 1.5
            },
            'pricing': {
                'max_price_change': 0.15,
                'min_margin': 0.2,
                'optimization_frequency': 'weekly'
            }
        }
    
    def render(self, current_config: Dict = None):
        """Render configuration interface."""
        if current_config is None:
            current_config = self.default_config
        
        st.subheader("Model Configuration")
        
        # Demand Forecast Configuration
        st.write("### Demand Forecast Settings")
        horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=90,
            value=current_config['demand_forecast']['horizon']
        )
        confidence = st.slider(
            "Confidence Level",
            min_value=0.8,
            max_value=0.99,
            value=current_config['demand_forecast']['confidence_level']
        )
        
        # Inventory Configuration
        st.write("### Inventory Settings")
        safety_stock = st.slider(
            "Safety Stock Level",
            min_value=0.8,
            max_value=0.99,
            value=current_config['inventory']['safety_stock_level']
        )
        lead_time = st.number_input(
            "Lead Time (days)",
            min_value=1,
            max_value=30,
            value=current_config['inventory']['lead_time_days']
        )
        
        # Pricing Configuration
        st.write("### Pricing Settings")
        max_price_change = st.slider(
            "Maximum Price Change",
            min_value=0.05,
            max_value=0.3,
            value=current_config['pricing']['max_price_change']
        )
        min_margin = st.slider(
            "Minimum Margin",
            min_value=0.1,
            max_value=0.4,
            value=current_config['pricing']['min_margin']
        )
        
        # Update configuration
        new_config = {
            'demand_forecast': {
                'horizon': horizon,
                'confidence_level': confidence,
                'update_frequency': current_config['demand_forecast']['update_frequency']
            },
            'inventory': {
                'safety_stock_level': safety_stock,
                'lead_time_days': lead_time,
                'reorder_point_multiplier': current_config['inventory']['reorder_point_multiplier']
            },
            'pricing': {
                'max_price_change': max_price_change,
                'min_margin': min_margin,
                'optimization_frequency': current_config['pricing']['optimization_frequency']
            }
        }
        
        if st.button("Save Configuration"):
            try:
                self.save_config(new_config)
                st.success("Configuration saved successfully!")
            except Exception as e:
                st.error(f"Error saving configuration: {str(e)}")
                self.logger.error(f"Configuration save error: {str(e)}")
        
        return new_config
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            raise
