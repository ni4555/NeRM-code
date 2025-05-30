import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic based on the maximum edge distance in each pair of nodes
    # We will calculate the heuristics based on the fact that a shorter edge is preferred
    # as it contributes less to the total distance of the tour.
    
    # Calculate the maximum distance for each edge
    max_edge_distances = np.max(distance_matrix, axis=0)
    
    # Calculate the minimum distance for each edge
    min_edge_distances = np.min(distance_matrix, axis=0)
    
    # The heuristic is a function of the difference between the maximum and minimum distances
    # This encourages the algorithm to avoid edges with a high range of possible distances
    heuristics = max_edge_distances - min_edge_distances
    
    return heuristics