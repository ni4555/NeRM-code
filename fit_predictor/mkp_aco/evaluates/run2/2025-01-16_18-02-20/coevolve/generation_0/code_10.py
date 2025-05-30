import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total potential prize per dimension
    total_potential_prize = np.sum(prize, axis=0)
    
    # Calculate the total weight per dimension
    total_weight = np.sum(weight, axis=0)
    
    # Calculate the normalized potential prize per dimension
    normalized_potential_prize = total_potential_prize / total_weight
    
    # Calculate the heuristics score for each item
    heuristics = normalized_potential_prize * np.prod(weight, axis=1)
    
    return heuristics