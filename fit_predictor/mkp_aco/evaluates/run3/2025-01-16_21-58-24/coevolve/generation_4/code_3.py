import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the weight for each dimension is fixed to 1
    # Normalize the weight to sum to 1 across each item
    weight_normalized = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Calculate the heuristic value for each item
    # We use the ratio of the prize to the normalized weight as the heuristic
    heuristics = prize / weight_normalized
    
    # Enforce the weight constraint by ensuring that the sum of weights
    # of selected items does not exceed the total weight capacity (which is 1 in this case)
    # We can simply normalize the prize by the weight sum to ensure that
    # items with higher weight are less likely to be selected if the prize is not proportionally higher
    heuristics /= np.sum(weight, axis=1, keepdims=True)
    
    return heuristics