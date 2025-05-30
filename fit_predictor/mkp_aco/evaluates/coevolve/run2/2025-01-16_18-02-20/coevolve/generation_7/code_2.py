import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight = prize / weight
    
    # Calculate the heuristic as the negative of the value-to-weight ratio
    # since we want to maximize the total prize, we minimize the negative ratio
    heuristics = -value_to_weight
    
    # Normalize the heuristics to ensure they are in a comparable range
    max_heuristic = np.max(heuristics)
    min_heuristic = np.min(heuristics)
    heuristics = (heuristics - min_heuristic) / (max_heuristic - min_heuristic)
    
    return heuristics