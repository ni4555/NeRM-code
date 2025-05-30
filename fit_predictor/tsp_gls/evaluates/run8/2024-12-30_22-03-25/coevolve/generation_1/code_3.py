import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric and the diagonal is filled with zeros
    # The heuristic can be based on the distance to the next nearest neighbor
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):  # No need to consider self-loops
            # Find the next nearest neighbor for node i excluding itself and the node j
            next_nearest_neighbors = distance_matrix[i].copy()
            next_nearest_neighbors[i] = float('inf')
            next_nearest_neighbors[j] = float('inf')
            next_nearest_neighbors = np.delete(next_nearest_neighbors, np.argmax(next_nearest_neighbors))
            next_nearest_neighbor_distance = np.argmin(next_nearest_neighbors)  # The index of the nearest neighbor
            
            # Calculate the heuristic value
            heuristics[i, j] = distance_matrix[i, next_nearest_neighbor_distance] + distance_matrix[next_nearest_neighbor_distance, j] - distance_matrix[i, j]
    
    # Return the heuristics matrix
    return heuristics