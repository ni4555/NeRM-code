import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    
    # Calculate the degree of each node
    degrees = np.sum(distance_matrix, axis=1)
    
    # Calculate the importance of each node based on its degree and the average
    # degree of its neighbors
    importance = degrees + np.mean(distance_matrix, axis=1)
    
    # Introduce a penalty for high degree nodes to balance the exploration
    penalty = np.maximum(0, 1 - degrees / (2 * n))
    
    # Define a function to calculate the local feature of each edge based on
    # the importance of its nodes, the distance between them, and the penalty
    def local_feature(i, j):
        return (1 / (importance[i] * importance[j])) * distance_matrix[i, j] * penalty[i] * penalty[j]
    
    # Calculate the heuristic matrix by applying the local feature function
    # to each edge
    heuristic_matrix = np.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic_matrix[i, j] = local_feature(i, j)
    
    # Normalize the heuristic matrix to the range [0, 1]
    max_feature = np.max(heuristic_matrix)
    if max_feature > 0:
        heuristic_matrix /= max_feature
    
    return heuristic_matrix
