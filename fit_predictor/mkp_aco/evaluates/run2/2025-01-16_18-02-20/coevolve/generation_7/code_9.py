import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to ensure they sum to 1 across all items
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Generate a random number for each item to introduce stochasticity
    random_numbers = np.random.rand(len(prize))
    
    # Calculate heuristics based on the normalized ratios and random numbers
    heuristics = normalized_ratio * random_numbers
    
    return heuristics