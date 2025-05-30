import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Set diagonal to -inf to avoid revisiting nodes
    np.fill_diagonal(heuristic_matrix, np.inf)
    
    # Fill the heuristic_matrix with the shortest edge from each node
    for i in range(distance_matrix.shape[0]):
        min_edges = np.argmin(distance_matrix[i, :])
        heuristic_matrix[i, min_edges] = distance_matrix[i, min_edges]
        # Set the opposite edge as the same cost to avoid symmetry
        heuristic_matrix[min_edges, i] = distance_matrix[i, min_edges]
    
    return heuristic_matrix