import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric and the diagonal elements are zero
    # Initialize a matrix of the same shape as the distance matrix with all values set to a very high number
    heuristics_matrix = np.full(distance_matrix.shape, np.inf)

    # Set the diagonal elements to zero as they are not considered in the heuristic
    np.fill_diagonal(heuristics_matrix, 0)

    # Compute the heuristic for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Avoid the diagonal (self-loops)
                # Assuming that the heuristic is inversely proportional to the distance
                heuristics_matrix[i][j] = 1 / distance_matrix[i][j]

    return heuristics_matrix