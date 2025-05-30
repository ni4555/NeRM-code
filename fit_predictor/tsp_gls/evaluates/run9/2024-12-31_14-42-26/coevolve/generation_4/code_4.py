import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array with the same shape as the distance matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the diagonal of the distance matrix
    np.fill_diagonal(heuristics, np.inf)
    
    # Calculate the minimum pairwise distances and their dynamic adjustments
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            min_pairwise_distance = distance_matrix[i][j]
            dynamic_adjustment = np.random.rand() * min_pairwise_distance  # Random adjustment for exploration
            heuristics[i][j] = heuristics[j][i] = min_pairwise_distance + dynamic_adjustment
    
    return heuristics