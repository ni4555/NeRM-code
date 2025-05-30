import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Calculate the priority based on the value-to-weight ratio
    priority = value_to_weight_ratio.sum(axis=1)
    
    # Normalize the priority to get a probability distribution
    probabilities = priority / priority.sum()
    
    # Sample items based on the probability distribution
    sample_indices = np.random.choice(range(n), size=int(m / 2), replace=False, p=probabilities)
    
    # Select items with the highest priority within the weight constraint
    heuristics = np.zeros(n)
    for index in sample_indices:
        heuristics[index] = 1.0
    
    return heuristics