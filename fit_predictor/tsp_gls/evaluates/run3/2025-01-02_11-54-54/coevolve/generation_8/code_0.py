import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix to store heuristics
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on the given distance matrix
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Recalibrate edge costs against the total graph cost
                heuristics_matrix[i][j] = distance_matrix[i][j] / np.sum(distance_matrix[i])
            else:
                # No heuristic for the diagonal elements (self-loops)
                heuristics_matrix[i][j] = 0.0
    
    return heuristics_matrix