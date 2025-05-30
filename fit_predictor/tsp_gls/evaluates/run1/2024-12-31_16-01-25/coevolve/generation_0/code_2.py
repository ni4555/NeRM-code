import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Get the number of nodes in the distance matrix
    num_nodes = distance_matrix.shape[0]
    
    # Initialize an array of zeros with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # For each edge, calculate the Manhattan distance and set it as the heuristic value
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Calculate the Manhattan distance
                manhattan_distance = np.abs(i - j)
                heuristics[i, j] = manhattan_distance
    
    return heuristics