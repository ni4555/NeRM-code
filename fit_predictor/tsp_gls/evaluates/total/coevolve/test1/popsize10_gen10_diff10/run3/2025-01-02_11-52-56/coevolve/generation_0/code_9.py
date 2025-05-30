import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as the distance matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Iterate over each row and column of the distance matrix to compute the heuristic
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # If the edge is not the diagonal (i.e., it's a real edge), set the heuristic
            if i != j:
                heuristics[i, j] = max(distance_matrix[i, j], distance_matrix[j, i])
            else:
                # For the diagonal elements, which represent the distance from a node to itself,
                # we can set the heuristic to a very large number to avoid including this edge
                heuristics[i, j] = float('inf')
    
    return heuristics