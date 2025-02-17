import logging
import streamlit as st
import pandas as pd
from components.data_upload import DataUploadComponent
from components.visualization import MetricsVisualizer
from components.configuration import ModelConfiguration

class Dashboard:
    def __init__(self):
        self.data_upload = DataUploadComponent()
        self.visualizer = MetricsVisualizer()
        self.configuration = ModelConfiguration()
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        st.title("Supply Chain Optimization Dashboard")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate to",
            ["Data Upload", "Demand Forecast", "Inventory", "Pricing", "Configuration"]
        )
        
        if page == "Data Upload":
            data = self.data_upload.render()
            if data is not None:
                st.session_state['data'] = data
        
        elif page == "Demand Forecast":
            if 'data' in st.session_state:
                self.visualizer.plot_demand_forecast(
                    st.session_state['data'],
                    self.get_forecast_data()
                )
            else:
                st.warning("Please upload data first!")
        
        elif page == "Inventory":
            if 'data' in st.session_state:
                self.visualizer.plot_inventory_levels(
                    self.get_inventory_data()
                )
            else:
                st.warning("Please upload data first!")
        
        elif page == "Pricing":
            if 'data' in st.session_state:
                self.visualizer.plot_price_optimization(
                    self.get_price_data()
                )
            else:
                st.warning("Please upload data first!")
        
        elif page == "Configuration":
            self.configuration.render()
    
    def get_forecast_data(self):
        # Placeholder for getting forecast data
        return pd.DataFrame()
    
    def get_inventory_data(self):
        # Placeholder for getting inventory data
        return pd.DataFrame()
    
    def get_price_data(self):
        # Placeholder for getting price data
        return pd.DataFrame()