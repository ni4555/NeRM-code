import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric, the diagonal elements are the distance to itself
    # and can be considered to have a cost of 0.
    num_nodes = distance_matrix.shape[0]
    np.fill_diagonal(distance_matrix, 0)
    
    # Calculate the total cost of the fully connected graph (all edges included)
    total_cost = np.sum(distance_matrix)
    
    # Initialize a matrix of the same shape as the distance matrix to hold the heuristics
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Iterate over each edge to compute heuristic
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                # The heuristic value is the edge cost divided by the total cost
                heuristics_matrix[i, j] = distance_matrix[i, j] / total_cost
    
    return heuristics_matrix