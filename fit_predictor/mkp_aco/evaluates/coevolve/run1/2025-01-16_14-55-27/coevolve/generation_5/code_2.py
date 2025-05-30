import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize weights to be in the same scale as prizes for comparison
    weight_normalized = weight / weight.sum(axis=1, keepdims=True)
    
    # Calculate the prize-to-weight ratio for each item
    prize_to_weight_ratio = prize / weight_normalized
    
    # Calculate the cumulative performance metrics (e.g., sum of ratios)
    cumulative_performance = np.cumsum(prize_to_weight_ratio, axis=0)
    
    # Apply adaptive stochastic sampling to prioritize items
    # This is a placeholder for a more complex stochastic sampling algorithm
    # Here we just shuffle the items randomly as an example
    shuffled_indices = np.random.permutation(cumulative_performance.shape[0])
    cumulative_performance = cumulative_performance[shuffled_indices]
    
    # Apply multi-criteria ranking system
    # Here we use cumulative performance as the sole criterion for simplicity
    heuristics = cumulative_performance
    
    return heuristics