import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios so that the highest ratio corresponds to the maximum value
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Convert the normalized ratios into a heuristic score
    # The score is inversely proportional to the normalized ratio to prioritize higher ratios
    heuristic = 1 / normalized_ratio
    
    return heuristic