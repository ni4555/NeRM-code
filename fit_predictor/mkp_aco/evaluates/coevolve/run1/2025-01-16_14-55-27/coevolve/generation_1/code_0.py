import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristics are based on the ratio of prize to weight for each item.
    # Each item's heuristic is computed as prize per unit weight for each dimension,
    # summed across dimensions, then the maximum of these sums for each item is taken.
    # This approach assumes that the constraint is fixed to 1 per dimension, as mentioned.
    
    # Compute the sum of prizes divided by weight for each dimension for each item
    dimension_heuristics = np.sum(prize / weight, axis=1)
    
    # Return the maximum heuristic for each item
    return np.max(dimension_heuristics, axis=0)