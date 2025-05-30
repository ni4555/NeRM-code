import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and the diagonal is filled with zeros
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Minimize the sum of the longest edges in each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                heuristics_matrix[i, j] = distance_matrix[i, j] - np.min(distance_matrix[i, :]) - np.min(distance_matrix[:, j])
    
    return heuristics_matrix