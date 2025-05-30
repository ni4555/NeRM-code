import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the diagonal of the distance matrix to exclude self-loops
    diagonal = np.diag(distance_matrix)
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic values
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                # Subtract the distance from i to j from the diagonal element of i to get the heuristic
                heuristics_matrix[i][j] = diagonal[i] - distance_matrix[i][j]
            else:
                # Set the diagonal elements to a large number to avoid including self-loops
                heuristics_matrix[i][j] = float('inf')
    
    return heuristics_matrix