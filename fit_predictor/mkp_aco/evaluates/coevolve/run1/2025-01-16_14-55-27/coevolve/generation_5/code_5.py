import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total value of each item
    total_value = np.sum(prize, axis=1)
    
    # Calculate the total weight of each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = total_value / total_weight
    
    # Integrate adaptive stochastic sampling and innovative heuristic algorithms
    # For simplicity, we use a random shuffle as a placeholder for adaptive stochastic sampling
    np.random.shuffle(value_to_weight_ratio)
    
    # Create a multi-criteria ranking system
    # We consider both value-to-weight ratio and prize value for ranking
    rank = value_to_weight_ratio * prize
    
    # Return the ranking as heuristics
    heuristics = rank / np.sum(rank)  # Normalize to sum to 1
    
    return heuristics