import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize row distances by subtracting the minimum distance in each row
    min_distances = np.min(distance_matrix, axis=1, keepdims=True)
    normalized_distances = distance_matrix - min_distances
    
    # Correlate with the graph's total cost (sum of all row distances)
    total_cost = np.sum(distance_matrix, axis=1)
    normalized_distances /= total_cost
    
    # Return the normalized distances as the heuristic values
    return normalized_distances