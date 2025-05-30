import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Euclidean distances
    euclidean_distances = np.sqrt(np.sum((distance_matrix - np.mean(distance_matrix, axis=1, keepdims=True)) ** 2, axis=2))
    
    # Calculate Chebyshev distances
    chebyshev_distances = np.max(distance_matrix, axis=1)
    
    # Combine the two distances using a simple linear weighting
    combined_distances = 0.5 * euclidean_distances + 0.5 * chebyshev_distances
    
    # Adjust the values to ensure they are negative (since we are trying to minimize the heuristic)
    heuristics = -combined_distances
    
    return heuristics