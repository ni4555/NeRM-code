import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Normalize the distance matrix
    row_sums = distance_matrix.sum(axis=1)
    col_sums = distance_matrix.sum(axis=0)
    row_max = row_sums.max()
    col_max = col_sums.max()
    
    # Distance-based normalization
    for i in range(n):
        for j in range(n):
            heuristics[i, j] = distance_matrix[i, j] / (row_max + col_max)
    
    # Dynamic minimum spanning tree construction
    # For simplicity, we will use a heuristic approach to simulate the effect
    # by considering the minimum edge for each node pair as the "spanning edge"
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristics[i, j] = min(heuristics[i, j], distance_matrix[i, j])
    
    return heuristics