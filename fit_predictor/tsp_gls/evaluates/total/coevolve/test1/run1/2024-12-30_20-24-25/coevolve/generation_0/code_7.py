import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The precision heuristic matrix will be based on the average distance
    # between each pair of nodes. Lower average distances between a pair suggest
    # that the edges connected to those nodes are less costly.
    # We'll calculate this heuristic for each edge and return it as the heuristic
    # value for that edge.
    num_nodes = distance_matrix.shape[0]
    precision_heuristic_matrix = np.zeros_like(distance_matrix)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Calculate the average distance between nodes i and j
                avg_distance = np.mean(distance_matrix[i, :]) + np.mean(distance_matrix[:, j])
                precision_heuristic_matrix[i, j] = avg_distance
    
    return precision_heuristic_matrix