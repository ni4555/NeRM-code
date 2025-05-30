import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic scores array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the prize-to-weight ratio for each item
    prize_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Calculate the cumulative performance metric for each item
    cumulative_performance = np.cumsum(prize_to_weight_ratio)
    
    # Normalize the cumulative performance to create a heuristic score
    heuristics = cumulative_performance / cumulative_performance[-1]
    
    return heuristics