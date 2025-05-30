import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric
    n = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)
    
    # Example heuristic: Invert the distance to get a heuristic
    # This encourages the algorithm to avoid large distances
    heuristics = 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero
    
    return heuristics