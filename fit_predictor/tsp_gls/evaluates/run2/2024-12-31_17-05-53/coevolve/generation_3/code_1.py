import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance between each pair of nodes
    # Since the distance matrix is symmetric, we only need to calculate half of it
    rows, cols = distance_matrix.shape
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    for i in range(rows):
        for j in range(i + 1, cols):
            # Calculate the Manhattan distance for the edge between nodes i and j
            heuristics_matrix[i, j] = heuristics_matrix[j, i] = np.abs(i - j).sum()
    
    return heuristics_matrix