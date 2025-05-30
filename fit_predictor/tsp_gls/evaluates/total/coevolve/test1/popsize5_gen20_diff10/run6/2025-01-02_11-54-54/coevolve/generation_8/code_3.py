import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix to store the heuristic values
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the total graph cost
    total_graph_cost = np.sum(distance_matrix)
    
    # Iterate over each pair of nodes to calculate the heuristic value
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Recalibrate edge cost against the total graph cost
                edge_cost = distance_matrix[i, j]
                # Calculate the heuristic value
                heuristic_value = edge_cost / total_graph_cost
                # Assign the heuristic value to the corresponding position in the heuristic matrix
                heuristic_matrix[i, j] = heuristic_value
    
    return heuristic_matrix