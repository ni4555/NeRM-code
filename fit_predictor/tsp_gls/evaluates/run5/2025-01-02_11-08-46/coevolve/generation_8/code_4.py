import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Advanced distance-based normalization techniques
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Robust minimum sum heuristic for precise edge selection
    # Assuming that a lower normalized distance indicates a better edge to include
    # in the solution, we take the minimum sum of the normalized distances for each edge
    min_sum_normalized_distances = np.min(normalized_distances, axis=0)
    
    # Return the prior indicators for each edge
    return min_sum_normalized_distances