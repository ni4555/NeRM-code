import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)

    # Calculate the maximum distance in the matrix
    max_distance = np.max(distance_matrix)

    # Iterate over each edge in the matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Exclude the diagonal
                # Calculate the heuristic for the edge (i, j)
                # This is a simple heuristic that assumes the edge with the smallest
                # distance is the most favorable to include in the solution
                heuristics_matrix[i, j] = max_distance - distance_matrix[i, j]

    return heuristics_matrix