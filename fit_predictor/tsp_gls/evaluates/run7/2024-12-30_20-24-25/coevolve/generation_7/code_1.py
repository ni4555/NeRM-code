import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Euclidean distance heuristic
    euclidean_heuristics = np.sum(distance_matrix, axis=1)
    
    # Calculate the Chebyshev distance heuristic
    chebyshev_heuristics = np.max(distance_matrix, axis=1)
    
    # Normalize both heuristics to the range [0, 1]
    max_possible_distance = np.max(distance_matrix)
    euclidean_heuristics /= max_possible_distance
    chebyshev_heuristics /= max_possible_distance
    
    # Combine the two heuristics using a simple average (this can be adjusted)
    combined_heuristics = (euclidean_heuristics + chebyshev_heuristics) / 2
    
    return combined_heuristics