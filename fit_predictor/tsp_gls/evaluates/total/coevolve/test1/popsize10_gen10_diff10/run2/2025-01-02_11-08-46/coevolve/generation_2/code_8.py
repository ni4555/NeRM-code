import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Invert the distance matrix (larger distances become smaller and vice versa)
    inverse_distance_matrix = 1.0 / (distance_matrix + 1e-10)  # Adding a small value to avoid division by zero
    
    # Normalize the inverted distance matrix to ensure all values are between 0 and 1
    min_value = np.min(inverse_distance_matrix)
    max_value = np.max(inverse_distance_matrix)
    normalized_matrix = (inverse_distance_matrix - min_value) / (max_value - min_value)
    
    return normalized_matrix