import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the distance-weighted normalization
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix)
    normalized_distances = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate the resilient minimum spanning tree heuristic
    # For simplicity, we use the minimum spanning tree algorithm which is a common heuristic
    # Note: In a real-world scenario, this would be replaced with a more sophisticated heuristic
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix
    
    # Convert the distance matrix to a sparse matrix
    distance_matrix_sparse = csr_matrix(distance_matrix)
    # Compute the minimum spanning tree
    mst = minimum_spanning_tree(distance_matrix_sparse)
    
    # Calculate the heuristic values
    heuristic_values = np.zeros_like(distance_matrix)
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if mst[i, j] != 0:  # If there is an edge between i and j in the MST
                heuristic_values[i, j] = normalized_distances[i, j]
            else:
                heuristic_values[i, j] = 1  # No edge, assign a high penalty
    
    return heuristic_values