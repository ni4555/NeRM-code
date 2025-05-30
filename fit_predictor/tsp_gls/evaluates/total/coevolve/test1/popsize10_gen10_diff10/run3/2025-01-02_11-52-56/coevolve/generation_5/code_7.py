import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the symmetric distance matrix to account for bidirectional edges
    symmetric_distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(symmetric_distance_matrix)
    
    # Apply a simple distance-based heuristic: the lower the distance, the less "bad" the edge
    heuristic_matrix = 1 / symmetric_distance_matrix
    
    # Refine the heuristic matrix to avoid infinite values and ensure non-negative values
    # Replace infinite values with a large number and non-positive values with a small number
    heuristic_matrix = np.nan_to_num(heuristic_matrix)
    heuristic_matrix[heuristic_matrix == np.inf] = 1e10
    heuristic_matrix[heuristic_matrix < 0] = 1e-10
    
    return heuristic_matrix