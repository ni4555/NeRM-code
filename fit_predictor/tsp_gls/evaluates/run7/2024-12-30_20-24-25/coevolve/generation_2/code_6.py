import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and the diagonal is filled with zeros
    # Initialize the heuristic matrix with the same shape as the distance_matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on the distance_matrix
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):  # since the matrix is symmetric, avoid redundant calculations
            # A simple heuristic could be the average distance to all other nodes from the node (i, j)
            # For example, we can use the average distance to all nodes except itself and the node it's being compared with
            if i != j:
                average_distance = np.mean(distance_matrix[i, ~np.isin(np.arange(len(distance_matrix)), [i, j])])
                heuristic_matrix[i, j] = average_distance
                heuristic_matrix[j, i] = average_distance
    
    return heuristic_matrix