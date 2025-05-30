import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # For the edge (i, j), the heuristic is the distance between i and j
            # Minus the minimum distance from either node to the nearest node not in the current path
            # This is a simplified heuristic for demonstration purposes
            heuristic_matrix[i, j] = distance_matrix[i, j]
            for k in range(len(distance_matrix)):
                if k != i and k != j:
                    # Calculate the minimum distance from k to either i or j
                    min_dist_k_to_i = min(distance_matrix[k, i], distance_matrix[i, k])
                    min_dist_k_to_j = min(distance_matrix[k, j], distance_matrix[j, k])
                    # Update the heuristic to include this cost
                    heuristic_matrix[i, j] += min_dist_k_to_i + min_dist_k_to_j
    
    return heuristic_matrix