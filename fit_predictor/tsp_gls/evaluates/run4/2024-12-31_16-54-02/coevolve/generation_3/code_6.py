import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic values based on the distance matrix
    # For simplicity, we'll use the reciprocal of the distances as the heuristic
    # Note: This is a placeholder heuristic and should be replaced with the actual heuristic
    # as described in the problem description.
    heuristic_matrix = 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    
    return heuristic_matrix