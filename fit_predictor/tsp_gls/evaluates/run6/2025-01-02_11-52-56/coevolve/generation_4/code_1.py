import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is a square matrix where
    # distance_matrix[i][j] represents the distance between city i and city j.
    # Here, we will use a simple heuristic where the cost of an edge is inversely proportional
    # to the distance between the cities, with a small constant adjustment to avoid zero cost edges.
    
    # Calculate the heuristic values as the inverse of the distance matrix
    # plus a small constant to avoid division by zero.
    heuristic_values = 1.0 / (distance_matrix + 1e-10)
    
    return heuristic_values