# main.py
import logging
import os
import json  # Add this import
from concurrent import futures
import grpc
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import streamlit as st
import threading
from typing import Dict

# Import our modules
from data.processors.data_processor import DataPreprocessor, FeatureEngineer
from models.demand_forecaster import DemandForecaster
from models.inventory_monitor import InventoryMonitor
from models.replenishment_optimizer import ReplenishmentOptimizer
from models.supplier_manager import SupplierManager
from models.price_optimizer import PriceOptimizer
from models.ensemble_model import EnsembleModel
from api.grpc.services import DemandForecastService, InventoryMonitorService
from dashboard.components.app import Dashboard

class SupplyChainSystem:
    def __init__(self):        
        self.logger = logging.getLogger(__name__)
        self.load_config()
        self.initialize_components()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.logger.warning("Config file not found, using defaults")
            self.config = {
                'model_params': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'learning_rate': 0.001
                },
                'api_config': {
                    'grpc_port': 50051,
                    'rest_port': 8000
                },
                'dashboard_config': {
                    'port': 8501
                }
            }
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize data processing
            self.data_processor = DataPreprocessor(self.config['model_params'])
            self.feature_engineer = FeatureEngineer(self.config['model_params'])
            
            # Initialize models
            self.demand_forecaster = DemandForecaster(self.config['model_params'])
            self.inventory_monitor = InventoryMonitor(self.config['model_params'])
            self.replenishment_optimizer = ReplenishmentOptimizer(self.config['model_params'])
            self.supplier_manager = SupplierManager(self.config['model_params'])
            self.price_optimizer = PriceOptimizer(self.config['model_params'])
            
            # Initialize ensemble
            self.ensemble = EnsembleModel([
                self.demand_forecaster.model,
                self.inventory_monitor.stockout_predictor
            ])
            
            # Initialize API servers
            self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            self.rest_app = FastAPI()
            
            # Initialize dashboard
            self.dashboard = Dashboard()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def start_grpc_server(self):
        """Start the gRPC server."""
        try:
            # Add services to the server
            DemandForecastService.add_to_server(self.grpc_server)
            InventoryMonitorService.add_to_server(self.grpc_server)
            
            # Start server
            self.grpc_server.add_insecure_port(f'[::]:{self.config["api_config"]["grpc_port"]}')
            self.grpc_server.start()
            self.logger.info(f"gRPC server started on port {self.config['api_config']['grpc_port']}")
            
        except Exception as e:
            self.logger.error(f"Error starting gRPC server: {str(e)}")
            raise
    
    def start_rest_server(self):
        """Start the REST API server."""
        try:
            # Add CORS middleware
            self.rest_app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Start server
            uvicorn.run(
                self.rest_app,
                host="0.0.0.0",
                port=self.config['api_config']['rest_port']
            )
            self.logger.info(f"REST server started on port {self.config['api_config']['rest_port']}")
            
        except Exception as e:
            self.logger.error(f"Error starting REST server: {str(e)}")
            raise
    
    def start_dashboard(self):
        """Start the Streamlit dashboard."""
        try:
            self.dashboard.run()
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {str(e)}")
            raise
    
    def run(self):
        """Run the entire system."""
        try:
            # Start servers in separate threads
            grpc_thread = threading.Thread(target=self.start_grpc_server)
            rest_thread = threading.Thread(target=self.start_rest_server)
            
            grpc_thread.start()
            rest_thread.start()
            
            # Start dashboard in main thread
            self.start_dashboard()
            
            # Wait for threads to complete
            grpc_thread.join()
            rest_thread.join()