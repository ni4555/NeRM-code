import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is square (n x n)
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    
    # Calculate the heuristic values using the inverse of the distances
    # Multiplying by a small constant to ensure non-zero values
    # This constant can be adjusted based on the expected range of the distances
    small_constant = 1e-10
    heuristic_values = (1 / (distance_matrix + small_constant)).astype(np.float64)
    
    return heuristic_values