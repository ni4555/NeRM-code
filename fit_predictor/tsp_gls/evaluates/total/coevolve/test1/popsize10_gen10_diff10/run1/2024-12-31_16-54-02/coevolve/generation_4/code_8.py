import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure that the distance_matrix is a square matrix
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "The distance matrix must be square."
    
    # Subtract the minimum distance to any node from the distance of each edge
    # This is a simple heuristic based on the minimum distance to any node
    min_row_sums = np.min(distance_matrix, axis=1)
    min_col_sums = np.min(distance_matrix, axis=0)
    # Calculate the heuristics by subtracting the minimum distance to any node
    heuristics = distance_matrix - np.minimum(min_row_sums[:, np.newaxis], min_col_sums[np.newaxis, :])
    
    return heuristics