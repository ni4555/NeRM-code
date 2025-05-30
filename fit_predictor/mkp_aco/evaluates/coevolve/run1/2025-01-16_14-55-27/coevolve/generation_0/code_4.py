import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total potential value for each item
    potential_value = prize * np.prod(weight, axis=1)
    
    # Normalize the potential value by the maximum value to scale the results
    max_potential = np.max(potential_value)
    normalized_potential = potential_value / max_potential
    
    # Calculate the heuristics by subtracting the normalized potential from 1
    heuristics = 1 - normalized_potential
    
    return heuristics