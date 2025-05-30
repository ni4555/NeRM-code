import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum pairwise distances
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create a new matrix with dynamic adjustments, for example, by subtracting a small constant
    # This is a simple way to introduce a dynamic adjustment to the heuristic
    dynamic_adjustment = np.full(distance_matrix.shape, 0.1)
    adjusted_min_distances = min_distances - dynamic_adjustment
    
    # Return the heuristic matrix which combines the minimum pairwise distances and the dynamic adjustments
    return adjusted_min_distances