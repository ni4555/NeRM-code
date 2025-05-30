import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is symmetric and the diagonal elements are zeros
    # Initialize the heuristic array with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the diagonal of the distance matrix to avoid considering the edge to the same node
    diagonal = np.arange(distance_matrix.shape[0])
    
    # Iterate over the rows to compute the heuristic for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # The heuristic value is the distance from the starting node to node i
                # plus the distance from node i to node j, minus the minimum distance
                # between node i and any other node, to avoid overestimating the cost
                heuristic_matrix[i, j] = distance_matrix[0, i] + distance_matrix[i, j] - np.min(distance_matrix[i, :])
    
    return heuristic_matrix