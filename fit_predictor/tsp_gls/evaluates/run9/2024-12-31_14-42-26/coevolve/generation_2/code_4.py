import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the same shape array with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic for each edge by applying some heuristic strategy
    # This is a placeholder for the actual heuristic computation
    # For demonstration purposes, let's use the simple average distance from the first city to all others
    # This is not an effective heuristic and should be replaced with a more sophisticated one
    heuristic_matrix[:, 0] = np.mean(distance_matrix[0, 1:])
    heuristic_matrix[0, :] = np.mean(distance_matrix[1:, 0])
    
    return heuristic_matrix