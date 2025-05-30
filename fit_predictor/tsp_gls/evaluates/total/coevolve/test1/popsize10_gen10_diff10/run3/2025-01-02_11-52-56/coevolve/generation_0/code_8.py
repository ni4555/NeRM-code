import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array with the same shape as the distance_matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Iterate over each pair of nodes to compute Manhattan distance
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # Compute Manhattan distance between nodes i and j
            # This assumes that the distance_matrix represents Manhattan distance
            manhattan_distance = abs(i - j)
            # Store the computed Manhattan distance in the heuristic matrix
            heuristics[i, j] = manhattan_distance
            heuristics[j, i] = manhattan_distance
    
    return heuristics