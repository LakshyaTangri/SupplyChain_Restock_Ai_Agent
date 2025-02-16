class PriceOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
    
    def analyze_price_elasticity(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price elasticity of demand for each product."""
        elasticities = {}
        
        for product in data['product_id'].unique():
            product_data = data[data['product_id'] == product]
            
            # Calculate percentage changes
            price_pct_change = product_data['price'].pct_change()
            demand_pct_change = product_data['units_sold'].pct_change()
            
            # Calculate elasticity
            valid_mask = (price_pct_change != 0) & (~price_pct_change.isna()) & (~demand_pct_change.isna())
            if valid_mask.any():
                elasticity = (demand_pct_change[valid_mask] / price_pct_change[valid_mask]).mean()
                elasticities[product] = elasticity
        
        return elasticities
    
    def optimize_prices(self, current_prices: pd.DataFrame, elasticities: Dict[str, float]) -> pd.DataFrame:
        """Optimize prices based on elasticity and constraints."""
        optimized_prices = current_prices.copy()
        
        for product, elasticity in elasticities.items():
            current_price = current_prices.loc[current_prices['product_id'] == product, 'price'].iloc[0]
            
            # If elasticity is elastic (< -1), consider lowering price
            if elasticity < -1:
                price_change = min(0.1, abs(1/elasticity))  # max 10% decrease
                new_price = current_price * (1 - price_change)
            # If elasticity is inelastic (> -1), consider raising price
            else:
                price_change = min(0.05, abs(1/elasticity))  # max 5% increase
                new_price = current_price * (1 + price_change)
            
            # Apply price constraints
            new_price = max(new_price, current_price * 0.8)  # max 20% decrease
            new_price = min(new_price, current_price * 1.2)  # max 20% increase
            
            optimized_prices.loc[optimized_prices['product_id'] == product, 'price'] = new_price
        
        return optimized_prices
