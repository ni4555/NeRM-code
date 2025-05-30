import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric and the diagonal is filled with zeros
    # We will calculate the minimum distance from each node to any other node excluding itself
    # This is the heuristic value for each edge (i, j) where i != j
    
    # Initialize an array with the same shape as the distance matrix with large values
    heuristics = np.full(distance_matrix.shape, np.inf)
    
    # Iterate over each node
    for i in range(distance_matrix.shape[0]):
        # Calculate the minimum distance to any other node
        min_distance = np.min(distance_matrix[i, :i] + distance_matrix[i, i+1:])
        # Update the heuristics array for the edges connected to node i
        heuristics[i, i+1:] = min_distance
        heuristics[i+1:, i] = min_distance  # Symmetry
    
    return heuristics