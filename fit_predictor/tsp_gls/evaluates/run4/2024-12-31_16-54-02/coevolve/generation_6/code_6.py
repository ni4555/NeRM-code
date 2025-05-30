import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # Compute the heuristic as the shortest path from i to j without revisiting nodes
            shortest_path = np.sort(distance_matrix[i])[:len(distance_matrix) - 1]  # Exclude the starting node
            heuristic = np.sum(shortest_path)
            heuristic_matrix[i][j] = heuristic
            heuristic_matrix[j][i] = heuristic  # Since the matrix is symmetric
    
    return heuristic_matrix