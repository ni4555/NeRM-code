import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristics to be a simple distance from each node to the nearest node in the matrix
    # This is a naive heuristic for demonstration purposes
    num_nodes = distance_matrix.shape[0]
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                min_distance = np.min(distance_matrix[i, ~np.isclose(distance_matrix[i], 0)])
                heuristic_matrix[i, j] = min_distance
    
    return heuristic_matrix