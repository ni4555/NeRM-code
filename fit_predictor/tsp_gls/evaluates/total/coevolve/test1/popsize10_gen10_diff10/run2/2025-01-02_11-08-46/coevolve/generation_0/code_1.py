import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function computes a simple heuristic for the TSP.
    # It uses the Manhattan distance between the nodes as an indicator.
    # The Manhattan distance between two nodes at positions (x1, y1) and (x2, y2) is given by:
    # dist = abs(x1 - x2) + abs(y1 - y2)
    # In a 2D distance matrix, the Manhattan distance can be approximated by the sum of the absolute differences
    # of their corresponding row and column indices.
    
    # Create a new matrix for the heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Iterate over the rows and columns of the distance matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Ignore the diagonal elements
                # Compute the Manhattan distance as the heuristic for this edge
                heuristics[i, j] = abs(i - j) + abs(i % distance_matrix.shape[0] - j % distance_matrix.shape[0])
    
    return heuristics