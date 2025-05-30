import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize distance matrix
    max_distance = np.max(distance_matrix)
    normalized_matrix = distance_matrix / max_distance
    
    # Compute minimum sum heuristic
    min_sum_heuristic = np.min(distance_matrix, axis=0)
    
    # Combine the normalized distances and minimum sum heuristic
    combined_heuristic = normalized_matrix + min_sum_heuristic
    
    # Apply a robust minimum sum heuristic for optimal edge selection
    robust_min_sum_heuristic = np.argmin(combined_heuristic, axis=1)
    
    # Adjust the combined heuristic values using the robust minimum sum heuristic
    adjusted_heuristic = combined_heuristic + robust_min_sum_heuristic - np.min(combined_heuristic)
    
    return adjusted_heuristic