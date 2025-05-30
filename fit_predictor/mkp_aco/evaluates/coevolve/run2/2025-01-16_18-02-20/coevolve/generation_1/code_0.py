import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    heuristics = np.zeros_like(prize)
    
    # Assuming that the weight constraint is 1 for each dimension,
    # we can calculate the heuristic by dividing the prize by the weight.
    # Since weight is of shape (n, m) and the constraint is 1 for each dimension,
    # we can take the maximum weight across all dimensions for each item.
    max_weight_per_item = np.max(weight, axis=1)
    
    # Calculate the heuristic for each item
    heuristics = prize / max_weight_per_item
    
    return heuristics