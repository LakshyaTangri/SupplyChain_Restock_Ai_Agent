from src.api.supply_chain_api import app  # Import the FastAPI app instance
from fastapi.testclient import TestClient
from src.utils.data_loader import load_inventory_data  # Import the data loading function
import json

# Create a test client for the API
client = TestClient(app)

# Load inventory data
file_path = "D:/Github/SupplyChain_Restock_Ai_Agent/Data/retail_store_inventory.csv"
inventory_data = load_inventory_data(file_path)

def forecast_demand(start_date: str, end_date: str):
    """Generate demand forecast"""
    forecast_response = client.post(
        "/forecast/demand",
        json={
            "start_date": start_date,
            "end_date": end_date
        }
    )
    print(forecast_response.json())

def optimize_inventory(inventory_data: list):
    """Optimize inventory"""
    inventory_response = client.post(
        "/optimize/inventory",
        json=inventory_data
    )
    print(inventory_response.json())

def optimize_supply_chain():
    """Get complete supply chain optimization"""
    optimization_response = client.post(
        "/optimize/supply-chain"
    )
    print(optimization_response.json())

if __name__ == "__main__":
    # Generate demand forecast
    forecast_demand("2024-02-16", "2024-03-16")

    # Optimize inventory
    inventory_data = [inventory_data.iloc[0].to_dict()]  # Example usage with the first row of data
    optimize_inventory(inventory_data)

    # Get complete supply chain optimization
    optimize_supply_chain()

    # Run the FastAPI app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)