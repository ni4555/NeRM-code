import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a square matrix where distance_matrix[i][j] is the distance between city i and city j.
    # We'll use Manhattan distance as our heuristic.
    
    # Calculate the Manhattan distance for each edge
    Manhattan_distances = np.abs(distance_matrix - np.transpose(distance_matrix))
    
    # The heuristic for each edge is the sum of the Manhattan distances
    # We subtract the distance from the matrix to avoid double counting the same edge
    heuristic_matrix = Manhattan_distances.sum(axis=1) - distance_matrix.diagonal()
    
    return heuristic_matrix