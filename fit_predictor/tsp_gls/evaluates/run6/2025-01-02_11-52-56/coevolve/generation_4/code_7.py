import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the same shape matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Iterate over the distance matrix to calculate the heuristic for each edge
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):  # No need to check the diagonal
            # Apply Manhattan distance heuristic
            # For simplicity, we're assuming the graph is undirected and the distance matrix is symmetric
            heuristic_matrix[i, j] = np.abs(i - j)  # This is equivalent to the Manhattan distance
            heuristic_matrix[j, i] = heuristic_matrix[i, j]  # Since the graph is undirected
    
    return heuristic_matrix

# Example usage:
# distance_matrix = np.array([[0, 2, 9], [1, 0, 6], [15, 7, 0]])
# heuristics = heuristics_v2(distance_matrix)
# print(heuristics)