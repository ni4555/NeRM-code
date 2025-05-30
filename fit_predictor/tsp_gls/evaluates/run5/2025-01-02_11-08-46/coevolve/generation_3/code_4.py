import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Distance-based normalization
    normalized_matrix = distance_matrix / np.max(distance_matrix)
    
    # Robust minimum sum heuristic
    min_sum = np.min(distance_matrix, axis=1)
    min_sum_matrix = np.tile(min_sum, (len(min_sum), 1))
    min_sum_matrix = np.abs(distance_matrix - min_sum_matrix)
    
    # Combine the two components
    combined_heuristics = normalized_matrix + min_sum_matrix
    
    # Apply a smoothing function to avoid extremely high heuristics
    smoothed_heuristics = np.clip(combined_heuristics, 0, 1)
    
    return smoothed_heuristics