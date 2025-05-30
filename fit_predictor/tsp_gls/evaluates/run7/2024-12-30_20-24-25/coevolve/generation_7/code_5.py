import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Euclidean distances and Chebyshev distances
    euclidean_distances = np.linalg.norm(distance_matrix, axis=1)
    chebyshev_distances = np.max(distance_matrix, axis=1)
    
    # Define a heuristic based on a weighted sum of Euclidean and Chebyshev distances
    # The weights can be adjusted based on the problem domain
    alpha = 0.5  # Weight for Euclidean distances
    beta = 0.5   # Weight for Chebyshev distances
    
    # Calculate the heuristic values
    heuristic_values = alpha * euclidean_distances + beta * chebyshev_distances
    
    # The negative of the heuristic values can be used to represent the "badness" of including an edge
    return -heuristic_values