import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # The heuristic is based on the assumption that shorter distances are preferred
    # This is a simple heuristic that assumes the lower the distance, the better the heuristic
    
    # Calculate the minimum distance for each pair of nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create a heuristic matrix where each cell indicates the penalty for including an edge
    # The lower the value, the better the edge
    heuristic_matrix = min_distances[:, np.newaxis] + min_distances - distance_matrix
    
    # The heuristic matrix is not penalizing the edges, but rather prioritizing them
    # since we want to minimize the total path length, we want to include the shortest edges first
    
    return heuristic_matrix