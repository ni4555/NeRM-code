import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the shortest path between any two nodes
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the shortest path from node i to node j
                # Assuming distance_matrix has the distance from i to j at distance_matrix[i][j]
                shortest_path = np.sort(distance_matrix[i])[1:]  # Exclude the distance to itself
                # Add the shortest path distances to the heuristic matrix
                for k in range(len(shortest_path)):
                    if k == 0:
                        heuristic_matrix[i][j] += shortest_path[k]
                    else:
                        heuristic_matrix[i][j] += shortest_path[k] - shortest_path[k-1]
    
    return heuristic_matrix