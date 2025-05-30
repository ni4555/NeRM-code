import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized value for each item
    normalized_value = prize / (np.sum(weight, axis=1) + 1e-6)
    
    # Calculate the heuristic based on weighted normalized value
    # Adding a small constant to avoid division by zero
    heuristic = normalized_value / np.sum(normalized_value) * len(prize)
    
    return heuristic