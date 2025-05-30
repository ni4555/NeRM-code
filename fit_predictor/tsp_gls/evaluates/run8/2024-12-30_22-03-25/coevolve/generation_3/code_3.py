import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with the same shape as the distance matrix
    heuristic_matrix = np.full_like(distance_matrix, np.inf)
    
    # Replace diagonal elements with zero since we don't want to include self-loops
    np.fill_diagonal(heuristic_matrix, 0)
    
    # Compute the heuristic by inverting the distance matrix
    # Note: If there are zero distances, this will result in division by zero.
    # You may want to handle such cases with a large penalty or by setting them to infinity.
    heuristic_matrix = 1.0 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    
    return heuristic_matrix