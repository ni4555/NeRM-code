import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and has at least one row and one column
    n = distance_matrix.shape[0]
    
    # Initialize the heuristics array with large values
    heuristics = np.full((n, n), np.inf)
    
    # Set diagonal elements to zero (no cost to visit the same city)
    np.fill_diagonal(heuristics, 0)
    
    # Calculate the minimum pairwise distances
    min_distances = np.min(distance_matrix, axis=1)
    
    # Adjust the heuristics based on the minimum distances
    heuristics = np.minimum(heuristics, min_distances)
    
    # Further adjust the heuristics by considering dynamic adjustments
    # This can be a placeholder for a more complex heuristic adjustment
    # For simplicity, we will just add a constant value (e.g., 0.1) to the minimum distances
    heuristics += 0.1
    
    return heuristics