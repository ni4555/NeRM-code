import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Distance-based normalization
    distance_matrix = distance_matrix / np.sum(distance_matrix, axis=0)

    # Robust minimum sum heuristic
    min_row_sums = np.sum(distance_matrix, axis=1)
    min_sum_heuristic = np.min(min_row_sums)
    min_row_sums = min_row_sums - min_sum_heuristic

    # Combine distance-based normalization with minimum sum heuristic
    combined_heuristic = distance_matrix + min_row_sums

    # Ensure the heuristic matrix has the same shape as the distance matrix
    return combined_heuristic.reshape(distance_matrix.shape)