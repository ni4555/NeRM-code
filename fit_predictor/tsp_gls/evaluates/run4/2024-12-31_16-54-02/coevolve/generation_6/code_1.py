import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with the same shape as the distance matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic values for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Calculate the shortest path between nodes i and j, avoiding the starting node
                # This is a simplified version of a heuristic; a more sophisticated approach might be needed
                # to achieve better performance.
                # For this example, we use the minimum distance from node i to all other nodes
                # and node j to all other nodes, excluding the start and end nodes themselves.
                heuristic_matrix[i][j] = min(distance_matrix[i][k] + distance_matrix[k][j] for k in range(len(distance_matrix)) if k != i and k != j)
    
    return heuristic_matrix