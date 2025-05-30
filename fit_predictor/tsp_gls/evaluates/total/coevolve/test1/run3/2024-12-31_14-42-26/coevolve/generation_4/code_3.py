import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum pairwise distances
    min_distances = np.min(distance_matrix, axis=1)
    
    # Apply a dynamic adjustment to the distances
    dynamic_adjustment = np.random.rand(distance_matrix.shape[0])
    
    # Combine the minimum distances and dynamic adjustments
    heuristics_values = min_distances + dynamic_adjustment
    
    return heuristics_values