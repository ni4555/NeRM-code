import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    num_nodes = distance_matrix.shape[0]
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Calculate the total graph cost as a reference
    total_graph_cost = np.sum(distance_matrix)
    
    # Calculate minimum distances per node
    min_distances = np.min(distance_matrix, axis=1)
    
    # Compute edge costs relative to the total graph cost
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Dynamic adjustment of heuristics based on minimum distances
                heuristics[i, j] = distance_matrix[i, j] / total_graph_cost + min_distances[i] / total_graph_cost
    
    return heuristics