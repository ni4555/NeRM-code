import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight of each item
    item_total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic for each item as the ratio of its prize to its total weight
    heuristics = prize / item_total_weight
    
    # Enforce the dimension constraint by scaling heuristics to sum to 1 across dimensions
    heuristics /= np.sum(heuristics, axis=1, keepdims=True)
    
    # Return the heuristics array
    return heuristics