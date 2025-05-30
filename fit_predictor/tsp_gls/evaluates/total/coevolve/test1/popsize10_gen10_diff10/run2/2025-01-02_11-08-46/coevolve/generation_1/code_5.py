import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Compute the initial tour cost
    initial_tour_cost = np.sum(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the cost of removing the edge (i, j)
            cost_without_edge = initial_tour_cost - distance_matrix[i, j] - distance_matrix[j, i]
            # Calculate the cost of adding the edge (i, k) followed by (k, j)
            # where k is a node that is not i or j
            for k in range(n):
                if k != i and k != j:
                    cost_with_new_edge = cost_without_edge + distance_matrix[i, k] + distance_matrix[k, j]
                    # Update the heuristic matrix
                    heuristic_matrix[i, j] = min(heuristic_matrix[i, j], cost_with_new_edge - initial_tour_cost)
    
    return heuristic_matrix