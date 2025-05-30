import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array to hold the heuristic values
    heuristic_values = np.zeros_like(distance_matrix, dtype=float)
    
    # Create a copy of the distance matrix to work with
    unvisited_matrix = distance_matrix.copy()
    
    # Find the minimum distance for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Find the minimum distance for the edge (i, j)
                min_distance = np.min(unvisited_matrix[i])
                
                # Set the heuristic value for this edge
                heuristic_values[i, j] = min_distance
                
                # Update the unvisited matrix for the next iteration
                unvisited_matrix[i] = np.delete(unvisited_matrix[i], j, axis=1)
    
    return heuristic_values