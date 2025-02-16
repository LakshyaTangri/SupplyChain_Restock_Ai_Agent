class ReplenishmentOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.q_table = {}
        self.logger = logging.getLogger(__name__)
    
    def get_state(self, inventory: float, demand: float) -> str:
        """Convert continuous state to discrete state for Q-learning."""
        inv_level = int(inventory / self.config['inventory_step'])
        demand_level = int(demand / self.config['demand_step'])
        return f"{inv_level}_{demand_level}"
    
    def get_action(self, state: str, epsilon: float = 0.1) -> int:
        """Get action using epsilon-greedy policy."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.config['n_actions'])
        
        if np.random.random() < epsilon:
            return np.random.randint(self.config['n_actions'])
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state: str, action: int, reward: float, next_state: str,
                      learning_rate: float = 0.1, discount_factor: float = 0.95):
        """Update Q-table using Q-learning algorithm."""
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.config['n_actions'])
        
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        self.q_table[state][action] = new_value