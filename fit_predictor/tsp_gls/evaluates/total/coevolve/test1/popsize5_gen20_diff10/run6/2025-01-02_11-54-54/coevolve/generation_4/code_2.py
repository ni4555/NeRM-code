import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that a higher heuristic value indicates a "worse" edge (i.e., one that should be avoided)
    # Initialize the heuristics array with the same shape as the distance matrix
    num_nodes = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on some heuristic function
    # Here we use a simple heuristic: the sum of the minimum distances to all other nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                heuristics[i, j] = np.sum(distance_matrix[i, :]) + np.sum(distance_matrix[:, j]) - distance_matrix[i, j]
            else:
                heuristics[i, j] = 0  # No heuristic for the diagonal elements
    
    return heuristics