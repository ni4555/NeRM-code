import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight of each item
    total_weight = np.sum(weight, axis=1)
    
    # Avoid division by zero if there are items with zero total weight
    with np.errstate(divide='ignore', invalid='ignore'):
        # Normalize prize by total weight to get the density
        density = np.true_divide(prize, total_weight)
        # Set the density to 0 where the total weight is 0
        density[np.isnan(density)] = 0
    
    return density