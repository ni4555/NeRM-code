import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Manhattan distances for each edge in the distance matrix
    # This heuristic assumes the graph is undirected
    heuristic_matrix = np.abs(distance_matrix - distance_matrix.T)
    
    # Normalize the heuristic matrix to get a better scale
    max_value = np.max(heuristic_matrix)
    min_value = np.min(heuristic_matrix)
    range_value = max_value - min_value
    normalized_matrix = (heuristic_matrix - min_value) / range_value
    
    return normalized_matrix