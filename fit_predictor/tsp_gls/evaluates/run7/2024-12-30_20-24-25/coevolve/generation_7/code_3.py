import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a 2D numpy array with shape (n, n)
    # where n is the number of nodes in the TSP problem
    
    # Compute Chebyshev distances
    chebyshev_distances = np.max(np.abs(distance_matrix), axis=1)
    
    # Compute Euclidean distances
    euclidean_distances = np.linalg.norm(distance_matrix, axis=1)
    
    # Combine Euclidean and Chebyshev distances
    # This is a simple heuristic, the exact formula would depend on the specifics of the algorithm
    # Here we just take the minimum of the two distances as a heuristic score
    heuristic_scores = np.minimum(euclidean_distances, chebyshev_distances)
    
    return heuristic_scores