import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Implementing a heuristic that assesses the minimum pairwise distances among nodes
    # One simple approach could be to calculate the average distance from each node to all other nodes
    num_nodes = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)
    
    for i in range(num_nodes):
        # Calculate the average distance from node i to all other nodes
        average_distance = np.mean(distance_matrix[i])
        heuristics[i] = average_distance
    
    return heuristics