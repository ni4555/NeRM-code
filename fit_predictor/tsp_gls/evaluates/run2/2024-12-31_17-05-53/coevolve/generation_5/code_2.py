import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance between each pair of nodes
    # The Manhattan distance between two points (i, j) and (k, l) is abs(i-k) + abs(j-l)
    # In a matrix form, this translates to the sum of the absolute differences of each dimension
    
    # Assuming the distance matrix is symmetric, we can use either the upper or lower triangle
    # to calculate the Manhattan distances.
    # We'll use the lower triangle to avoid redundant calculations.
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1):  # Exclude the diagonal and upper triangle
            heuristic_matrix[i, j] = np.abs(i - j).sum()
    
    return heuristic_matrix