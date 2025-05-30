import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a 2D array to store the heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the Manhattan distance to estimate the cost of including each edge
    # The Manhattan distance is the sum of the absolute differences of their Cartesian coordinates
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            # Calculate Manhattan distance between node i and node j
            manhattan_distance = np.sum(np.abs(distance_matrix[i] - distance_matrix[j]))
            # The heuristic is the negative of the Manhattan distance
            # because we want to maximize the heuristic value (which will correspond to a lower cost)
            heuristics[i][j] = -manhattan_distance
            heuristics[j][i] = -manhattan_distance  # The matrix is symmetric
    
    return heuristics