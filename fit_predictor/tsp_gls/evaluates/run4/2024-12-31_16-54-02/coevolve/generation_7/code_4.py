import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as the distance_matrix with zeros
    heuristics = np.zeros_like(distance_matrix)

    # Iterate over each pair of nodes (i, j)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristics value, which is the distance from the node to all other nodes
                # minus the distance from the node to node j (to avoid counting it twice).
                heuristics[i][j] = np.sum(distance_matrix[i]) - distance_matrix[i][j]
            else:
                # For the diagonal elements (self-loops), we set the heuristics to a large number
                # or to a value that signifies that this edge should not be considered.
                heuristics[i][j] = float('inf')

    return heuristics