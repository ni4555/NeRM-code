import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the profit-to-weight ratio for each item
    profit_to_weight_ratio = prize / weight
    
    # Normalize the ratio to ensure that the heuristic is on a comparable scale
    normalized_ratio = profit_to_weight_ratio / np.sum(profit_to_weight_ratio)
    
    # Adjust the normalized ratio based on compliance with the knapsack constraints
    # Assuming that the sum of weights for each item is less than or equal to 1 for each dimension
    # We can use the normalized ratio directly as the heuristic value
    heuristics = normalized_ratio
    
    return heuristics