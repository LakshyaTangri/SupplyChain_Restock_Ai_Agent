# SupplyChain_Restock_Ai_Agent

A small-scale version of McDonald's AI model and data used to optimize its supply chain.

## Project Overview

This project implements a simplified AI-driven supply chain optimization system, inspired by McDonald's approach. It focuses on demand forecasting, inventory management, and stock replenishment for a subset of McDonald's menu items.

## Features

- **Demand Forecasting**: Predicts future product demand based on historical sales data.
- **Inventory Monitoring**: Tracks real-time inventory levels across locations.
- **Stock Replenishment**: Automates restocking decisions based on forecasts and current inventory.
- **Supplier Management**: Optimizes supplier selection and order placement.
- **Dynamic Pricing**: Adjusts prices based on demand and inventory levels.
- **Ensemble Models**: Combines multiple AI models for improved accuracy.

## Project Structure

```
├── data/
│   ├── historical_sales.csv
│   └── inventory_levels.csv
├── src/
│   ├── demand_forecasting.py
│   ├── inventory_monitoring.py
│   ├── stock_replenishment.py
│   ├── supplier_management.py
│   ├── dynamic_pricing.py
│   └── ensemble_models.py
├── models/
│   ├── base_models/
│   │   ├── model1.pkl
│   │   ├── model2.pkl
│   │   └── model3.pkl
│   └── ensemble_model.pkl
├── tests/
│   ├── test_forecasting.py
│   └── test_ensemble.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SupplyChain_Restock_Ai_Agent.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Place historical sales data in `data/historical_sales.csv`
   - Update current inventory levels in `data/inventory_levels.csv`

2. Run the main script:
   ```
   python src/ensemble_models.py
   ```

3. View the results in the console output or generated reports.

## Testing

Run the test suite using:
```
python -m unittest discover tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
