import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio by subtracting the minimum ratio
    normalized_ratio = value_to_weight_ratio - np.min(value_to_weight_ratio)
    
    # Rank the items based on the normalized ratio
    ranking = np.argsort(normalized_ratio)[::-1]
    
    # Calculate the heuristic score for each item
    # The heuristic score is the rank multiplied by the normalized ratio
    heuristics = normalized_ratio[ranking] * (ranking + 1)
    
    return heuristics