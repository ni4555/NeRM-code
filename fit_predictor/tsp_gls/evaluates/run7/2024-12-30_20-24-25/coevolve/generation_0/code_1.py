import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the center of the matrix
    matrix_size = distance_matrix.shape[0]
    center_index = matrix_size // 2
    if matrix_size % 2 == 0:
        center_row = center_index
        center_col = center_index
    else:
        center_row = center_index
        center_col = center_index + 1
    
    center = distance_matrix[center_row, center_col]

    # Compute the heuristic matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            if i != j:
                # Calculate the distance from the edge to the center
                distance_to_center = (distance_matrix[i, j] + distance_matrix[j, i]) / 2
                # Estimate the "badness" of the edge by its distance from the center
                heuristic = 1 / (1 + distance_to_center / center)
                heuristic_matrix[i, j] = heuristic
                if i != j:
                    heuristic_matrix[j, i] = heuristic  # For symmetric matrices

    return heuristic_matrix