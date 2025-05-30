import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the total graph cost
    total_cost = np.sum(distance_matrix)
    
    # Calculate the minimum distance for each node
    min_distances = np.min(distance_matrix, axis=1)
    
    # Compute edge costs relative to the total graph cost
    edge_costs = distance_matrix / total_cost
    
    # Adjust heuristics based on minimum distances per node
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics[i, j] = edge_costs[i, j] - (min_distances[i] / total_cost)
    
    return heuristics