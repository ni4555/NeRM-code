import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Implement your advanced distance-based normalization techniques and minimum sum heuristic here.
    # This is a placeholder for the actual implementation:
    # 1. Normalize the distance matrix.
    # 2. Apply the minimum sum heuristic.
    # 3. Return the heuristic values.
    
    # Placeholder values, to be replaced with actual heuristic calculations
    normalized_distance_matrix = distance_matrix / np.sum(distance_matrix)
    min_sum_heuristic = np.min(distance_matrix, axis=0)
    
    return min_sum_heuristic