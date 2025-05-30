import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the row-wise and column-wise sums
    row_sums = np.sum(distance_matrix, axis=1)
    col_sums = np.sum(distance_matrix, axis=0)
    
    # Normalize the row sums to create a base for the heuristic
    normalized_row_sums = row_sums / np.sum(row_sums)
    
    # Initialize the heuristic matrix with large values
    heuristic_matrix = np.full(distance_matrix.shape, np.inf)
    
    # Adjust the diagonal elements to be zero as the distance to the city itself is zero
    np.fill_diagonal(heuristic_matrix, 0)
    
    # Combine the row sums with the normalized sums to calculate the heuristic
    # The heuristic is the weighted sum of row and column sums for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristic_matrix[i][j] = normalized_row_sums[i] * col_sums[j]
    
    return heuristic_matrix