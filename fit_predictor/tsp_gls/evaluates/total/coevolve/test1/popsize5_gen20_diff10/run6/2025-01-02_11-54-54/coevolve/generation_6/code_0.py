import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the overall cost of the graph
    overall_cost = np.sum(distance_matrix)
    
    # Initialize an empty cost matrix with the same shape as the distance matrix
    cost_matrix = np.zeros_like(distance_matrix)
    
    # For each row in the distance matrix, subtract the minimum distance
    for i in range(distance_matrix.shape[0]):
        min_distance_in_row = np.min(distance_matrix[i])
        distance_normalized = distance_matrix[i] - min_distance_in_row
        cost_matrix[i] = distance_normalized
    
    # Normalize the cost matrix by the overall cost to ensure it is on a similar scale
    cost_matrix /= overall_cost
    
    return cost_matrix