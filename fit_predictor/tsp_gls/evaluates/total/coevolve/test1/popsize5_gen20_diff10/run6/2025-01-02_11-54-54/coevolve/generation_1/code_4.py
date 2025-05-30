import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is square and symmetric
    num_nodes = distance_matrix.shape[0]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the sum of distances for each edge
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # We do not consider the diagonal (self-loops)
                heuristics[i, j] = distance_matrix[i, j]
    
    return heuristics