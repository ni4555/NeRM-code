import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Compute Manhattan distance heuristics
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Use Manhattan distance to estimate the heuristic
                heuristics[i, j] = np.abs(i - j) * np.mean(distance_matrix)

    return heuristics