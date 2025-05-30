import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as the distance matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate Euclidean distances
    euclidean_distances = np.sqrt(np.sum((distance_matrix**2), axis=1))
    
    # Calculate Chebyshev distances
    chebyshev_distances = np.max(distance_matrix, axis=1)
    
    # The heuristic is a combination of both Euclidean and Chebyshev distances
    # Here, we use a simple linear combination as an example, but the actual heuristic
    # might be more complex depending on the specific requirements.
    heuristics = euclidean_distances + chebyshev_distances
    
    return heuristics