import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a square matrix of size n x n where n is the number of nodes
    n = distance_matrix.shape[0]

    # Initialize a matrix of the same shape as the input distance matrix
    heuristics_matrix = np.zeros_like(distance_matrix)

    # Implementing a distance-based normalization
    # Calculate the minimum spanning tree (MST) to use as a base for the heuristic
    # For simplicity, we'll use the Prim's algorithm for the MST calculation
    # (Note: In practice, the MST should be dynamically computed as the algorithm evolves)
    mst = np.zeros(n)
    visited = np.zeros(n, dtype=bool)
    mst[0] = 1
    for i in range(1, n):
        min_dist = np.inf
        min_idx = -1
        for j in range(n):
            if not visited[j] and distance_matrix[mst[i-1], j] < min_dist:
                min_dist = distance_matrix[mst[i-1], j]
                min_idx = j
        visited[min_idx] = True
        mst[i] = min_idx

    # Calculate the total distance of the MST
    mst_total_distance = np.sum(distance_matrix[mst[:-1], mst[1:]])

    # Calculate the heuristic for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # Normalize the edge weight based on the MST
                heuristics_matrix[i, j] = distance_matrix[i, j] / mst_total_distance

    return heuristics_matrix