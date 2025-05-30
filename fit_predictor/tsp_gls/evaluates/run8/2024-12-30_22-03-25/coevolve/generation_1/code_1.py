import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and the diagonal elements are 0
    # Calculate the minimum distances for each edge from the starting node (index 0)
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create an array to hold the heuristics for each edge
    heuristics = np.zeros_like(distance_matrix)
    
    # For each edge, calculate the heuristic by subtracting the minimum distance
    # from the current distance. If the minimum distance is 0 (the edge is the
    # starting edge), set the heuristic to a large number (e.g., np.inf)
    heuristics[distance_matrix != 0] = distance_matrix[distance_matrix != 0] - min_distances[distance_matrix != 0]
    heuristics[distance_matrix == 0] = np.inf
    
    return heuristics