import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix, which contains the distance from each node to itself
    diagonal = np.diag(distance_matrix)
    
    # Create a matrix of all ones with the same shape as the distance matrix
    ones_matrix = np.ones_like(distance_matrix)
    
    # Calculate the "badness" of including each edge by subtracting the diagonal from the sum of the corresponding row and column
    badness_matrix = ones_matrix - diagonal - distance_matrix
    
    # Normalize the badness matrix by dividing by the maximum value in the matrix
    # This step helps to ensure that the values in the output are within a manageable range
    normalized_badness_matrix = badness_matrix / np.max(badness_matrix)
    
    # The heuristics function returns the normalized badness matrix
    return normalized_badness_matrix