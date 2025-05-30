import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a square matrix with distances between nodes
    num_nodes = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute the total graph cost
    total_cost = np.sum(distance_matrix)
    
    # For each edge, compute its cost relative to the total graph cost
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_cost = distance_matrix[i, j]
            heuristics_matrix[i, j] = edge_cost / total_cost
            heuristics_matrix[j, i] = edge_cost / total_cost
    
    # Dynamically adjust heuristics based on minimum distances per node
    for i in range(num_nodes):
        min_distances = np.min(distance_matrix[i, :])
        heuristics_matrix[i, :] = heuristics_matrix[i, :] * (1 - min_distances / total_cost)
        heuristics_matrix[:, i] = heuristics_matrix[:, i] * (1 - min_distances / total_cost)
    
    return heuristics_matrix