import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix to store the minimum distances from the MST
    min_distances = np.full(distance_matrix.shape, np.inf)
    min_distances[0, 1] = distance_matrix[0, 1]  # Start with the first edge
    # Array to track which nodes have been included in the MST
    in_mst = np.zeros(distance_matrix.shape[0], dtype=bool)
    in_mst[0] = True

    # Number of nodes not yet included in the MST
    remaining_nodes = distance_matrix.shape[0] - 1

    # While there are nodes not yet included in the MST
    while remaining_nodes > 0:
        # Find the edge with the smallest weight that connects a node in the MST to a node not in the MST
        edge_indices = np.where(min_distances > 0)
        min_edge_value = np.min(min_distances[edge_indices[1]] * min_distances[edge_indices[0]])
        edge_from = edge_indices[0][np.argmin(min_distances[edge_indices[1]] * min_distances[edge_indices[0]])]
        edge_to = edge_indices[1][np.argmin(min_distances[edge_indices[1]] * min_distances[edge_indices[0]])]

        # Include the edge in the MST
        min_distances[edge_from, :] = 0
        min_distances[:, edge_from] = 0
        min_distances[edge_to, :] = 0
        min_distances[:, edge_to] = 0

        # Update the minimum distances
        for i in range(distance_matrix.shape[0]):
            if not in_mst[i] and distance_matrix[i, edge_from] < min_distances[edge_from, i]:
                min_distances[edge_from, i] = distance_matrix[i, edge_from]
            if not in_mst[i] and distance_matrix[i, edge_to] < min_distances[edge_to, i]:
                min_distances[edge_to, i] = distance_matrix[i, edge_to]

        # Mark the new node as included in the MST
        in_mst[edge_to] = True
        # Update the number of remaining nodes
        remaining_nodes -= 1

    # Create the heuristic matrix
    heuristic_matrix = np.where(min_distances == np.inf, 0, min_distances)
    return heuristic_matrix