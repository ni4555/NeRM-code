import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance is the heuristic function to be used
    # Manhattan distance between two points (i, j) and (k, l) is given by:
    # Manhattan(i, j, k, l) = |i - k| + |j - l|
    # This is calculated as the sum of the absolute differences of their respective coordinates.

    # Initialize a new matrix to store the heuristic values
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=np.float64)

    # Calculate Manhattan distance for each edge and fill the heuristic matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Skip the diagonal since distance from a node to itself is zero
                heuristic_matrix[i][j] = abs(i - j)

    return heuristic_matrix