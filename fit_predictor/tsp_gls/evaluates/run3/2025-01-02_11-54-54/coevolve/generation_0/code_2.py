import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics by subtracting the minimum distance for each row from the row's distances
    for i in range(distance_matrix.shape[0]):
        heuristics_matrix[i] = distance_matrix[i] - np.min(distance_matrix[i])
    
    return heuristics_matrix