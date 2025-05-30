import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array with the same shape as distance_matrix to store the heuristics
    heuristics = np.zeros_like(distance_matrix)

    # Compute the shortest path between all pairs of nodes using Dijkstra's algorithm
    for i in range(len(distance_matrix)):
        shortest_paths = np.sort(distance_matrix[i])[1:]  # Exclude the zero distance to the node itself
        # Calculate the heuristic as the difference between the shortest path and the next shortest path
        heuristics[i] = shortest_paths[0] - shortest_paths[1]
    
    return heuristics