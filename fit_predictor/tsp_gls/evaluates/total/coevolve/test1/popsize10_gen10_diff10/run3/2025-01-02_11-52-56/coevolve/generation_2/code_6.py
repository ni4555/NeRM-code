import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # The heuristic will be the sum of the distances to the nearest neighbor for each node
    # Initialize a heuristic array with the same shape as the distance matrix
    heuristic = np.zeros_like(distance_matrix)
    
    # Iterate over each node to compute the heuristic
    for i in range(distance_matrix.shape[0]):
        # For each node, find the nearest neighbor
        nearest_neighbor_index = np.argmin(distance_matrix[i, :])
        # Set the heuristic for this node to the distance to its nearest neighbor
        heuristic[i, nearest_neighbor_index] = distance_matrix[i, nearest_neighbor_index]
    
    return heuristic