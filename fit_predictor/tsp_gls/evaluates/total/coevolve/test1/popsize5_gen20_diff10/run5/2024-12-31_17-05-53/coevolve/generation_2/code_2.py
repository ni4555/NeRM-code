import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix
    np.fill_diagonal(distance_matrix, np.inf)
    
    # Compute the minimum distances between each pair of nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Calculate the heuristics for each edge
    heuristics = distance_matrix - min_distances[:, np.newaxis]
    
    return heuristics