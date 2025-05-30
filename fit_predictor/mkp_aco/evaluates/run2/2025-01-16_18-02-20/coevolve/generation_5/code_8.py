import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio to create a heuristic score
    # The normalization is done by dividing each ratio by the sum of all ratios
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # The heuristic score is simply the normalized ratio
    # which represents how promising it is to include the item in the solution
    heuristics = normalized_ratio
    
    return heuristics