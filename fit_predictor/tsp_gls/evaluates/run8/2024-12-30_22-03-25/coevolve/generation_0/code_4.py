import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Create an array of the same shape as the distance matrix
    # Initialize with a value that is high, so that we can find the minimum later
    heuristics = np.full(distance_matrix.shape, np.inf)

    # Iterate over each edge in the matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            # Exclude the diagonal and the edge from the start node to itself
            if i != j and i != 0:
                heuristics[i, j] = distance_matrix[i, j]
                
    return heuristics