import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that each item can only be included once and each dimension weight is 1,
    # the value-to-weight ratio is simply prize[i] (since weight[i] = 1 for all i)
    
    # Calculate the value-to-weight ratio
    value_to_weight_ratio = prize / weight
    
    # Return the array of value-to-weight ratios
    return value_to_weight_ratio