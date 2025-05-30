import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the prize array is of shape (n,) and the weight array is of shape (n, m)
    # where m is the dimension of weights for each item
    
    # Calculate the value/weight ratio for each item in each dimension
    value_weight_ratio = prize / weight
    
    # Calculate the average value/weight ratio across all dimensions for each item
    avg_ratio = np.mean(value_weight_ratio, axis=1)
    
    # Normalize the average value/weight ratio to get a heuristic value for each item
    max_ratio = np.max(avg_ratio)
    min_ratio = np.min(avg_ratio)
    heuristics = 2 * (avg_ratio - min_ratio) / (max_ratio - min_ratio) - 1
    
    return heuristics