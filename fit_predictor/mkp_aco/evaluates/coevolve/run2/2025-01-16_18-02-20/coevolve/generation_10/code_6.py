import numpy as np
import numpy as np
from scipy.stats import wasserstein_distance

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Normalize the value-to-weight ratios to ensure a proper comparison across items
    # Here we use min-max normalization, which scales the values to the range [0, 1]
    min_vtw = np.min(value_to_weight)
    max_vtw = np.max(value_to_weight)
    normalized_vtw = (value_to_weight - min_vtw) / (max_vtw - min_vtw)
    
    # Create a heuristic score based on the normalized value-to-weight ratios
    # The heuristic is simply the normalized value-to-weight ratio, which serves as a score
    heuristics = normalized_vtw
    
    return heuristics