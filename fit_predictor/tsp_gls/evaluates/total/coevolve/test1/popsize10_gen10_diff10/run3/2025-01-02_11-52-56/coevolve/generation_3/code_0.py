import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix to exclude the distance from a city to itself
    diag = np.diag(distance_matrix)
    
    # Create a matrix to store the heuristic values
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic value for each edge (i, j)
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # The heuristic value is the distance to the next city plus the average of the remaining distances
            # Note: This is a simple example, and the actual heuristic function can be more complex
            heuristic_matrix[i, j] = distance_matrix[i, j] + (diag[j] - distance_matrix[i, j]) / 2
            heuristic_matrix[j, i] = heuristic_matrix[i, j]  # The matrix is symmetric
    
    return heuristic_matrix