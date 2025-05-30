import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics matrix with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # Set diagonal elements to a large number (since we don't want to include the edge to the starting node)
    np.fill_diagonal(heuristics, np.inf)
    
    # Set the heuristics to the distance of the edge if it exists
    heuristics[distance_matrix < np.inf] = distance_matrix[distance_matrix < np.inf]
    
    return heuristics