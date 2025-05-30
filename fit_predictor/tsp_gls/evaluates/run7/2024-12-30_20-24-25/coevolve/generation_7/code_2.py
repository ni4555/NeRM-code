import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a new matrix with the same shape as the input distance matrix
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate Euclidean distance heuristic
    heuristics_matrix += np.sqrt(np.sum(np.square(distance_matrix), axis=1))
    
    # Calculate Chebyshev distance heuristic
    heuristics_matrix += np.max(np.abs(distance_matrix), axis=1)
    
    # Normalize the heuristics to ensure the values are non-negative and within a certain range
    heuristics_matrix = np.abs(heuristics_matrix)
    
    return heuristics_matrix