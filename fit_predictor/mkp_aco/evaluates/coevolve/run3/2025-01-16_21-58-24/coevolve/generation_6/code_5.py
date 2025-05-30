import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the minimum weight for each item across all dimensions
    min_weight_per_item = np.min(weight, axis=1)
    
    # Calculate the heuristic value for each item as the ratio of prize to minimum weight
    # This heuristic assumes that items with a higher prize-to-weight ratio are more promising
    heuristics = prize / min_weight_per_item
    
    # Return the heuristics array
    return heuristics