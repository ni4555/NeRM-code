import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the shortest path between any two nodes
    # using a modified Dijkstra's algorithm that does not revisit nodes.
    n = distance_matrix.shape[0]
    min_distances = np.full(n, np.inf)
    visited = np.zeros(n, dtype=bool)
    min_distances[0] = 0

    for _ in range(n):
        min_dist_node = np.where(min_distances == np.min(min_distances))[0][0]
        visited[min_dist_node] = True

        for i in range(n):
            if not visited[i] and distance_matrix[min_dist_node, i] < min_distances[i]:
                min_distances[i] = distance_matrix[min_dist_node, i]

    # Create a matrix with negative of the distances as heuristics
    heuristics_matrix = -min_distances
    return heuristics_matrix