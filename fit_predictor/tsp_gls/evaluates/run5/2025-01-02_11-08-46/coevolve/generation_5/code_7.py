import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Implement your distance-weighted normalization heuristic here
    # For example, let's assume the heuristic is the inverse of the distance
    # (This is just a placeholder, the actual heuristic would depend on the problem specifics)
    heuristics = 1.0 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Implement any additional logic required for the heuristic here
    
    return heuristics