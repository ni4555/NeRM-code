import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # For each pair of nodes (i, j), calculate the heuristics value
    # Assuming that the distance matrix is symmetric and the diagonal is filled with zeros
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Avoid the diagonal
                # Use a simple heuristic, e.g., the average distance to all other nodes
                # except the current node and its neighbors
                neighbors = np.where(distance_matrix[i, :] > 0)[0]
                non_neighbors = np.setdiff1d(range(distance_matrix.shape[0]), neighbors)
                non_neighbors.remove(i)
                if non_neighbors:
                    heuristics[i, j] = np.mean(distance_matrix[i, non_neighbors])
    
    return heuristics