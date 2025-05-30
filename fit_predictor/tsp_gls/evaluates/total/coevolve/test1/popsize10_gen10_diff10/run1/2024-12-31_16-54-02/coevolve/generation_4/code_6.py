import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance_matrix with large values
    heuristics_matrix = np.full(distance_matrix.shape, np.inf)
    
    # Set diagonal elements to 0, as we can't traverse the same node twice in TSP
    np.fill_diagonal(heuristics_matrix, 0)
    
    # Implement a simple heuristic: the higher the distance, the "worse" the edge
    heuristics_matrix = distance_matrix
    
    # Apply a discount factor to the heuristics, to give a more balanced perspective
    discount_factor = 0.5
    heuristics_matrix *= discount_factor
    
    return heuristics_matrix