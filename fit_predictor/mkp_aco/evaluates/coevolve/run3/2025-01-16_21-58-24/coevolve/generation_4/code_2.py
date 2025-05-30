import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic value for each item
    # Here we use a simple heuristic based on the ratio of prize to weight
    # This heuristic assumes that the constraint for each dimension is 1
    heuristic = prize / total_weight
    
    # Normalize the heuristic values to ensure they sum to 1
    heuristic /= np.sum(heuristic)
    
    return heuristic