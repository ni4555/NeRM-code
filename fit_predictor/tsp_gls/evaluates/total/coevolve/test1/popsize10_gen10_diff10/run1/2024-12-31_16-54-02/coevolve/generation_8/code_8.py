import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics matrix with the same shape as the distance matrix
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the shortest path between any two nodes using the Floyd-Warshall algorithm
    # This will be used to compute the heuristics values
    floyd_warshall_result = np.linalg.matrix_power(distance_matrix, 2)
    floyd_warshall_result = np.maximum(floyd_warshall_result, distance_matrix)
    floyd_warshall_result += np.transpose(distance_matrix)
    
    # Set the heuristics for each edge as the distance from the origin node to the destination node
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics_matrix[i][j] = floyd_warshall_result[i][j]
    
    return heuristics_matrix