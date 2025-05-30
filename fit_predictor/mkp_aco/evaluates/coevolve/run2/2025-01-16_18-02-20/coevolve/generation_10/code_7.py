import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Normalize the value-to-weight ratios to get a probability distribution
    total_ratio = np.sum(value_to_weight)
    probabilities = value_to_weight / total_ratio
    
    # Create a heuristics array that reflects the probability of including each item
    heuristics = probabilities
    
    return heuristics