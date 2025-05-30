import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix, which contains the distances to each city from itself
    diagonal = np.diag(distance_matrix)
    
    # Subtract each city's distance to itself from the corresponding distance matrix values
    # This gives us the edge weights without the diagonal elements
    edge_weights = distance_matrix - diagonal
    
    # Use the absolute values of the edge weights to get the prior indicators
    heuristics = np.abs(edge_weights)
    
    return heuristics