import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic based on the maximum distance to a nearest neighbor
    num_nodes = distance_matrix.shape[0]
    heuristic_values = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Calculate the heuristic for each edge
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Find the minimum distance to any other node for the current node i
                min_distance = np.min(distance_matrix[i])
                # The heuristic for edge (i, j) is the maximum distance to any node from i
                # (which would be the distance to the farthest node from i if i was the start of the tour)
                heuristic_values[i, j] = np.max(distance_matrix[i])
    
    return heuristic_values