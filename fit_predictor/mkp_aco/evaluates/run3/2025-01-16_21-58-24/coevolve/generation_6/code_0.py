import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = weight.shape
    heuristic_values = np.zeros(n)
    
    # Calculate the normalized profit and weight for each item
    normalized_profit = prize / weight.sum(axis=1)
    normalized_weight = weight.sum(axis=1) / weight.sum()
    
    # Incorporate a stochastic element to the heuristic by adding noise
    noise = np.random.normal(0, 0.1, size=n)
    
    # Combine normalized profit and weight to get the heuristic value
    # The heuristic can be adjusted by giving more weight to profit or weight constraints
    heuristic_values = normalized_profit * 0.8 + normalized_weight * 0.2 + noise
    
    # Normalize the heuristic values to ensure they sum to 1 for each item
    heuristic_values /= heuristic_values.sum()
    
    return heuristic_values