import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # We create a lower triangular matrix to avoid redundant calculations
    lower_triangle = distance_matrix[0:, 0:]
    
    # Initialize the heuristics matrix with zeros
    heuristics = np.zeros_like(lower_triangle)
    
    # Calculate the heuristics for each edge
    for i in range(lower_triangle.shape[0]):
        for j in range(i + 1, lower_triangle.shape[1]):
            # The heuristic for edge (i, j) is the distance from i to j
            heuristics[i, j] = lower_triangle[i, j]
    
    return heuristics