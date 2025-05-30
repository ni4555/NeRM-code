import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with the same shape as distance_matrix to store the heuristics
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the shortest path from each node to every other node
    for i in range(distance_matrix.shape[0]):
        # Create a copy of the distance matrix for the current iteration
        current_distance_matrix = np.copy(distance_matrix)
        
        # Set the distance from the current node to itself to infinity
        np.fill_diagonal(current_distance_matrix, np.inf)
        
        # Compute the shortest path from node i to all other nodes
        shortest_paths = np.linalg.mminus1(current_distance_matrix[i])
        
        # Update the heuristics matrix for node i based on the shortest paths
        for j in range(distance_matrix.shape[0]):
            if i != j:
                heuristics_matrix[i][j] = shortest_paths[j]
    
    return heuristics_matrix