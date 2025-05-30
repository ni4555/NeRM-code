import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Check if the distance matrix is square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square (n x n).")

    # Calculate the minimum spanning tree using Kruskal's algorithm
    # For simplicity, we'll use a brute-force approach to find the MST
    # in the distance matrix. This is not the most efficient method, but
    # it serves as an example of how to apply a heuristic based on MST.
    
    # Find the minimum edge for each node
    min_edges = np.min(distance_matrix, axis=1)
    
    # Initialize the MST and the heuristic array
    mst = np.zeros(distance_matrix.shape)
    heuristic = np.zeros(distance_matrix.shape)
    
    # Set the distance to itself to be 0 in the MST
    np.fill_diagonal(mst, 0)
    
    # Fill the MST with the minimum edges
    for i in range(distance_matrix.shape[0]):
        # Find the minimum edge not already in the MST
        for j in range(distance_matrix.shape[0]):
            if i != j and distance_matrix[i, j] not in mst:
                min_edge = np.min(distance_matrix[i, :])
                min_edge_index = np.where(distance_matrix[i, :] == min_edge)[0]
                mst[i, min_edge_index] = min_edge
                mst[min_edge_index, i] = min_edge
                break

    # Calculate the heuristic based on the MST
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            if i != j and distance_matrix[i, j] in mst[i, :]:
                heuristic[i, j] = -np.min(distance_matrix[i, :])
            else:
                heuristic[i, j] = np.inf
    
    return heuristic