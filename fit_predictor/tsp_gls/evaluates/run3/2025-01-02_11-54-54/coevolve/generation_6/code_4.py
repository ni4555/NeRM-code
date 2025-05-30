import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance for each row
    min_distance = np.min(distance_matrix, axis=1)
    
    # Normalize row distances by subtracting the minimum distance
    normalized_distance = distance_matrix - min_distance[:, np.newaxis]
    
    # Use the normalized distances as the heuristic values
    return normalized_distance