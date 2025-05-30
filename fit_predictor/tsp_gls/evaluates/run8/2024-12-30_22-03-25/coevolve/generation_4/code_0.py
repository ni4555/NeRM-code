import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a square matrix where distance_matrix[i][j] is the distance from node i to node j
    num_nodes = distance_matrix.shape[0]
    
    # Initialize an array to store the heuristics values
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the minimum distance from each node to all others
    for i in range(num_nodes):
        min_distance = np.min(distance_matrix[i])
        # Calculate the sum of the longest edges in each node pair
        longest_edges = np.max(distance_matrix[i])
        # Calculate the heuristics value
        heuristics[i] = longest_edges - min_distance
    
    return heuristics