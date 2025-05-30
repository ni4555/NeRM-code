import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Innovative heuristic to refine the distance matrix
    refined_matrix = distance_matrix * 0.9  # Example: normalize distances

    # Advanced edge-based heuristics (example: sum of distances to all other nodes)
    edge_heuristics = np.sum(refined_matrix, axis=0)

    # Distance normalization (example: divide by the maximum edge heuristic value)
    max_edge_heuristic = np.max(edge_heuristics)
    normalized_edge_heuristics = edge_heuristics / max_edge_heuristic

    # Optimized minimum sum heuristic (example: minimum sum of heuristics for each edge)
    min_sum_heuristic = np.min(normalized_edge_heuristics)

    # Apply the optimized minimum sum heuristic to the refined matrix
    for i in range(len(refined_matrix)):
        for j in range(len(refined_matrix[i])):
            refined_matrix[i][j] += min_sum_heuristic

    # Create a matrix with the heuristics that indicates how bad it is to include each edge
    heuristics_matrix = refined_matrix - distance_matrix

    return heuristics_matrix