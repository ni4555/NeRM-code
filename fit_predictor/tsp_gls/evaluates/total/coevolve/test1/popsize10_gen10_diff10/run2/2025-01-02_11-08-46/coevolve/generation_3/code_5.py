import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the row-wise and column-wise minima for each node
    row_minima = np.min(distance_matrix, axis=1)
    col_minima = np.min(distance_matrix, axis=0)
    
    # Compute the normalized distance for each edge
    normalized_distances = distance_matrix / (row_minima[:, np.newaxis] + col_minima[np.newaxis, :])
    
    # Apply the minimum sum heuristic by subtracting the minima from the normalized distances
    heuristic_values = normalized_distances - np.min(normalized_distances)
    
    return heuristic_values