import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal elements are 0
    # Create a copy of the distance matrix to avoid modifying the original
    heuristics_matrix = np.copy(distance_matrix)
    
    # Replace the diagonal elements with a very large number to avoid them being considered
    np.fill_diagonal(heuristics_matrix, np.inf)
    
    # Compute the heuristics as the sum of the distances of the edges
    # The heuristic for an edge (i, j) is the sum of the distances from i to all other nodes except j
    for i in range(len(heuristics_matrix)):
        for j in range(len(heuristics_matrix[i])):
            if i != j:
                heuristics_matrix[i][j] = np.sum(heuristics_matrix[i]) - heuristics_matrix[i][j]
    
    return heuristics_matrix