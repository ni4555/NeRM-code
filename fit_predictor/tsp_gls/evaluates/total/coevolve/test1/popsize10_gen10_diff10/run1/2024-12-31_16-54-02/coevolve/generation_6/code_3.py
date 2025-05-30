import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # For each pair of nodes, calculate the heuristic as the inverse of the distance
    # We use inverse because we are aiming to minimize the heuristic
    np.fill_diagonal(heuristic_matrix, np.inf)  # We can't include the same node twice
    heuristic_matrix = 1 / distance_matrix
    
    # Replace the infinities with a large number (or just zero if you prefer)
    # so that the heuristic does not affect the diagonal
    np.fill_diagonal(heuristic_matrix, 0)
    
    # In case there are zero distances, set their inverses to a large number to avoid division by zero
    np.where(distance_matrix == 0, np.inf, heuristic_matrix)
    
    return heuristic_matrix