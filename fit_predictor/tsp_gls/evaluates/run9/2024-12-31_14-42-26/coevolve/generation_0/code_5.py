import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix
    # This will be used to avoid considering the distance from a node to itself
    diagonal = np.diag(distance_matrix)
    
    # Subtract the diagonal from the distance matrix to get the non-diagonal elements
    non_diagonal = distance_matrix - diagonal
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-6
    
    # Compute the heuristic for each edge
    # The heuristic is based on the minimum pairwise distances among nodes
    heuristics = 1 / (non_diagonal / epsilon)
    
    # Return the heuristics matrix
    return heuristics