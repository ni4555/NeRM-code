import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate profit-to-weight ratio for each item
    profit_to_weight = prize / weight
    
    # Normalize the ratios so that they sum up to 1
    total_profit_to_weight = np.sum(profit_to_weight)
    if total_profit_to_weight == 0:
        # Handle the case where total profit to weight is zero to avoid division by zero
        return np.zeros_like(prize)
    
    normalized_profit_to_weight = profit_to_weight / total_profit_to_weight
    
    # The resulting normalized ratios can be considered as the heuristic value
    # for each item. The higher the value, the more promising the item is to include.
    return normalized_profit_to_weight