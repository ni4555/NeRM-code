import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with large values (infinity)
    num_nodes = distance_matrix.shape[0]
    heuristics = np.full((num_nodes, num_nodes), np.inf)

    # Calculate the shortest path for each pair of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Compute the shortest path between node i and node j
                # Assuming the input distance matrix is symmetric, we can use the Floyd-Warshall algorithm
                # as a simplified shortest path algorithm for demonstration purposes.
                # In practice, a more efficient algorithm like Dijkstra's should be used.
                for k in range(num_nodes):
                    # Update the shortest path if a shorter one is found
                    heuristics[i, j] = min(heuristics[i, j], distance_matrix[i, k] + distance_matrix[k, j])

    # Set the diagonal to zero as the distance from a node to itself is zero
    np.fill_diagonal(heuristics, 0)

    return heuristics