import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Calculate the heuristic as the logarithm of the ratio, which is a common heuristic
    heuristics = np.log(value_to_weight_ratio + 1e-10)  # Adding a small value to avoid log(0)
    
    return heuristics