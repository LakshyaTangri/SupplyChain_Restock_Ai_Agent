import grpc
from concurrent import futures
import logging
from typing import Dict, List
import numpy as np
from api.grpc.protos import supply_chain_pb2

class DemandForecastService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

    def GetForecast(self, request, context):
        try:
            predictions = self.model_manager.get_forecast(
                product_id=request.product_id,
                horizon=request.horizon,
                features=np.array(request.features)
            )
            
            return supply_chain_pb2.ForecastResponse(
                predictions=predictions.tolist(),
                confidence=0.95  # Example confidence value
            )
        except Exception as e:
            self.logger.error(f"Error in GetForecast: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return supply_chain_pb2.ForecastResponse()

    def TrainModel(self, request, context):
        try:
            performance = self.model_manager.train_model(
                training_data=request.training_data,
                model_type=request.model_type
            )
            
            return supply_chain_pb2.TrainingResponse(
                status="success",
                performance_metric=performance
            )
        except Exception as e:
            self.logger.error(f"Error in TrainModel: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return supply_chain_pb2.TrainingResponse(status="failed")

class InventoryMonitorService:
    def __init__(self, inventory_manager):
        self.inventory_manager = inventory_manager
        self.logger = logging.getLogger(__name__)

    def GetInventoryLevels(self, request, context):
        try:
            inventories = self.inventory_manager.get_inventory_levels(
                store_id=request.store_id,
                product_ids=request.product_ids
            )
            
            return supply_chain_pb2.InventoryResponse(
                inventories=[
                    supply_chain_pb2.ProductInventory(
                        product_id=inv['product_id'],
                        current_level=inv['current_level'],
                        safety_stock=inv['safety_stock'],
                        reorder_point=inv['reorder_point']
                    ) for inv in inventories
                ]
            )
        except Exception as e:
            self.logger.error(f"Error in GetInventoryLevels: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return supply_chain_pb2.InventoryResponse()