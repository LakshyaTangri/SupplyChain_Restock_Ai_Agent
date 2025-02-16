app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

app = FastAPI()

class ForecastRequest(BaseModel):
    product_id: str
    horizon: int
    features: List[float]

class InventoryUpdate(BaseModel):
    store_id: str
    product_id: str
    quantity: int
    transaction_type: str

class OptimizationRequest(BaseModel):
    store_id: str
    product_ids: List[str]
    optimization_type: str
    constraints: Optional[Dict]

@app.post("/api/v1/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        forecast = demand_forecaster.get_forecast(
            product_id=request.product_id,
            horizon=request.horizon,
            features=request.features
        )
        return {
            "status": "success",
            "predictions": forecast.tolist(),
            "confidence": 0.95
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/inventory/update")
async def update_inventory(update: InventoryUpdate):
    try:
        result = inventory_manager.update_inventory(
            store_id=update.store_id,
            product_id=update.product_id,
            quantity=update.quantity,
            transaction_type=update.transaction_type
        )
        return {
            "status": "success",
            "message": "Inventory updated successfully",
            "new_level": result['new_level']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/optimize")
async def optimize_parameters(request: OptimizationRequest):
    try:
        optimization_result = optimizer.optimize(
            store_id=request.store_id,
            product_ids=request.product_ids,
            optimization_type=request.optimization_type,
            constraints=request.constraints
        )
        return {
            "status": "success",
            "optimized_parameters": optimization_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/upload/training-data")
async def upload_training_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Validate and process the uploaded data
        validation_result = data_validator.validate(df)
        if not validation_result['valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data format: {validation_result['errors']}"
            )
        
        # Process the data and update models
        training_result = model_manager.train_with_new_data(df)
        
        return {
            "status": "success",
            "message": "Training data processed successfully",
            "metrics": training_result['metrics']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
