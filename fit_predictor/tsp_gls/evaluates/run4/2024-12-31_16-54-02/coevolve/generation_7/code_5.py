import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric (i.e., distance[i][j] == distance[j][i])
    # Initialize a matrix with the same shape as distance_matrix with large values
    heuristics_matrix = np.full(distance_matrix.shape, np.inf)
    
    # Calculate the shortest path between any two nodes
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:  # Exclude the edge to itself
                # Find the shortest path from node i to j
                # Here we're just using the given distance_matrix to simulate a shortest path
                heuristics_matrix[i][j] = distance_matrix[i][j]
    
    # Ensure that the distance from a node to itself is zero
    np.fill_diagonal(heuristics_matrix, 0)
    
    return heuristics_matrix