import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to make them suitable for stochastic sampling
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Create a heuristic array where each element is the probability of including the item
    # We use a simple heuristic where items with higher normalized ratio have a higher chance of being selected
    heuristics = np.random.rand(len(prize))
    heuristics /= heuristics.sum()  # Normalize to ensure probabilities sum to 1
    
    # Adjust heuristics based on the normalized value-to-weight ratio
    heuristics *= normalized_ratio
    
    # Normalize the adjusted heuristics again to ensure they sum to 1
    heuristics /= heuristics.sum()
    
    return heuristics

# Example usage:
# n = 5
# m = 1
# prize = np.array([60, 100, 120, 130, 140])
# weight = np.array([[1], [1], [1], [1], [1]])
# print(heuristics_v2(prize, weight))