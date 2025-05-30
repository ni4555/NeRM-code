import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total capacity
    total_capacity = weight.sum(axis=1)
    
    # Calculate the normalized weight for each item for each dimension
    normalized_weight = weight / total_capacity[:, np.newaxis]
    
    # Calculate the expected profit for each item for each dimension
    expected_profit = prize * normalized_weight
    
    # Calculate the heuristic for each item
    heuristic = expected_profit.sum(axis=1)
    
    return heuristic
