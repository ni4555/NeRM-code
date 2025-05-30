import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the total value for each item
    total_value = np.sum(prize, axis=1)
    
    # Normalize the total value by the total weight to get a density measure
    density = total_value / total_weight
    
    # Normalize the density to the range [0, 1]
    max_density = np.max(density)
    min_density = np.min(density)
    normalized_density = (density - min_density) / (max_density - min_density)
    
    # The heuristics is the normalized density, which indicates the promise of each item
    heuristics = normalized_density
    
    return heuristics