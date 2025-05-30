import numpy as np
import numpy as np
from scipy.stats import multinomial

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize values by their sum to get a relative importance
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = normalized_prize / weight
    
    # Calculate the heuristic for each item based on the value-to-weight ratio
    # Here we are simply taking the inverse of the value-to-weight ratio as a heuristic
    heuristics = 1 / value_to_weight_ratio
    
    # Normalize the heuristics to ensure they sum to 1
    heuristics /= np.sum(heuristics)
    
    return heuristics