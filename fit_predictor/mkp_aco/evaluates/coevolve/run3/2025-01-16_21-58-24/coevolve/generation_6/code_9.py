import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.size
    m = weight.shape[1]
    
    # Normalize the weights based on the sum of each item's weight vector
    norm_weight = weight / weight.sum(axis=1, keepdims=True)
    
    # Compute the normalized profit for each item
    norm_profit = prize / prize.sum()
    
    # Create a matrix that compares each dimension's normalized weight with normalized profit
    weight_profit_ratio = norm_weight / norm_profit
    
    # The heuristic value is the sum of all dimension ratios for each item
    heuristics = weight_profit_ratio.sum(axis=1)
    
    return heuristics