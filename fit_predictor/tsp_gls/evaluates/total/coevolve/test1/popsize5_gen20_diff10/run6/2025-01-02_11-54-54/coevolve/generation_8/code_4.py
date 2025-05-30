import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array of the same shape as the distance_matrix with large values
    heuristics = np.full(distance_matrix.shape, np.inf)
    
    # Calculate the diagonal elements (self-loops) as 0
    np.fill_diagonal(heuristics, 0)
    
    # Recalibrate edge costs against the total graph cost
    total_cost = np.sum(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics[i, j] = distance_matrix[i, j] / total_cost
    
    return heuristics