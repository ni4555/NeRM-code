import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the sum of the longest edges in each node pair
    max_edges_sum = np.sum(np.max(distance_matrix, axis=1), axis=0)
    
    # Calculate the minimum distance from each node to all others
    min_distances = np.min(distance_matrix, axis=1)
    
    # Combine the heuristics: max_edges_sum + sum of min_distances
    heuristic_values = max_edges_sum + np.sum(min_distances)
    
    # Create the heuristics matrix
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Fill the matrix with the calculated heuristics values
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                heuristics_matrix[i, j] = heuristic_values[i]
    
    return heuristics_matrix