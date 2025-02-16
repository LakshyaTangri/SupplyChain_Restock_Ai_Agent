import pandas as pd

def load_inventory_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess inventory data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed inventory data.
    """
    # Load the CSV file
    data = pd.read_csv('data/retail_store_inventory.csv')
    df = pd.DataFrame(data)
    # Extract relevant columns
    inventory_levels = data["Inventory Level"]
    units_sold = data["Units Sold"]

    # Perform any necessary preprocessing
    # For example, handling missing values
    data = data.dropna(subset=["Inventory Level", "Units Sold"])

    # Convert data types if necessary
    data["Inventory Level"] = data["Inventory Level"].astype(int)
    data["Units Sold"] = data["Units Sold"].astype(int)

    return data

# Example usage
if __name__ == "__main__":
    file_path = "D:/Github/SupplyChain_Restock_Ai_Agent/Data/retail_store_inventory.csv"
    data = load_inventory_data(file_path)
    print(data.head())