import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Iterate over each pair of nodes (i, j) to compute the heuristic values
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic value as the distance from node i to node j
                # plus the minimum distance from node j to any other node (excluding i)
                min_distance_to_other = np.min(distance_matrix[j, :i] + distance_matrix[j, i+1:])
                heuristic_value = distance_matrix[i, j] + min_distance_to_other
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix