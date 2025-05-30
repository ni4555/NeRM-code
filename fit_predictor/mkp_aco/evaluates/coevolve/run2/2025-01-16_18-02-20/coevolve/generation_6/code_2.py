import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Assuming a simple heuristic where we use the value-to-weight ratio
    # since the problem description did not specify how to calculate the heuristics.
    # This is just a placeholder for a more complex heuristic algorithm.
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio by subtracting the global minimum
    # and dividing by the global range to get a value between 0 and 1.
    normalized_ratio = (value_to_weight_ratio - np.min(value_to_weight_ratio)) / (np.max(value_to_weight_ratio) - np.min(value_to_weight_ratio))
    
    # Return the normalized ratio as the heuristic.
    return normalized_ratio