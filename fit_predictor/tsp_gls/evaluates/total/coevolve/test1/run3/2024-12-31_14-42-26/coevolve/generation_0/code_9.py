import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum pairwise distances among nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create a matrix where each entry is the inverse of the minimum distance
    # to its corresponding node (the smaller the distance, the larger the heuristic value)
    heuristic_matrix = 1 / min_distances
    
    # Replace any zero values with a large number (to avoid division by zero)
    # Zero values would occur if two nodes are the same (which should not happen in a proper distance matrix)
    heuristic_matrix[heuristic_matrix == 0] = np.inf
    
    return heuristic_matrix