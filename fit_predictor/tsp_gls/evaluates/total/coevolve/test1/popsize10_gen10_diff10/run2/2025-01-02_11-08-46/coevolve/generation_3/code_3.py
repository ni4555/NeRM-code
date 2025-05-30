import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix by subtracting the minimum distance from each edge
    min_distance = np.min(distance_matrix)
    normalized_matrix = distance_matrix - min_distance
    
    # Apply the minimum sum heuristic: return the sum of the smallest distances for each vertex
    min_sum_heuristic = np.min(normalized_matrix, axis=1)
    
    # The heuristic value for each edge is the sum of its two endpoints' heuristic values
    edge_heuristics = min_sum_heuristic + min_sum_heuristic[:, np.newaxis]
    
    return edge_heuristics