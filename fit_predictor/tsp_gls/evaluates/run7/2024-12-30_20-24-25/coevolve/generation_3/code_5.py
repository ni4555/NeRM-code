import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # We'll use the Chebyshev distance to calculate the heuristic for each edge
    # as a proxy for how "bad" it is to include an edge in a solution.
    # The Chebyshev distance is the maximum absolute difference in any dimension.
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the Chebyshev distance for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristic_matrix[i][j] = np.max([abs(distance_matrix[i][j]), 
                                                abs(distance_matrix[j][i])])
    
    return heuristic_matrix