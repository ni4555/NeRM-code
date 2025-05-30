import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as distance_matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on the distance matrix
    # For simplicity, let's use a distance-based heuristic: the reciprocal of the distance
    # This is a naive heuristic and may not be optimal for the given framework
    heuristics_matrix = 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    
    return heuristics_matrix