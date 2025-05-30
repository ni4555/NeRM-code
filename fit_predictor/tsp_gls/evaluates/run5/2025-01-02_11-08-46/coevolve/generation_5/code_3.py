import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute the sum of each row in the distance matrix
    row_sums = np.sum(distance_matrix, axis=1)
    
    # Normalize each element in the distance matrix by its row sum
    normalized_matrix = distance_matrix / row_sums[:, np.newaxis]
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-8
    normalized_matrix = np.clip(normalized_matrix, epsilon, 1 - epsilon)
    
    # Compute the heuristics matrix by subtracting the normalized values from 1
    heuristics_matrix = 1 - normalized_matrix
    
    return heuristics_matrix