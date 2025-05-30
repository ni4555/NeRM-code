import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Create a deep copy of the distance matrix to avoid modifying the original
    distance_matrix_copy = np.copy(distance_matrix)
    
    # Apply the Floyd-Warshall algorithm to find the shortest paths between all pairs of nodes
    np.fill_diagonal(distance_matrix_copy, np.inf)  # Set the distance from a node to itself to infinity
    np.fill_diagonal(distance_matrix_copy, 0)       # Set the distance from a node to itself to 0
    np.fill_lower(distance_matrix_copy, np.inf)     # Set the distance from a node to itself to infinity
    
    # Perform the Floyd-Warshall algorithm
    np.all(np.isfinite(distance_matrix_copy), axis=0, out=distance_matrix_copy)  # Check for negative cycles and set them to infinity
    np.all(np.isfinite(distance_matrix_copy), axis=1, out=distance_matrix_copy)  # Check for negative cycles and set them to infinity
    np.fill_diagonal(distance_matrix_copy, 0)       # Reset the diagonal to 0
    
    np.fill_diagonal(distance_matrix_copy, np.inf)  # Set the distance from a node to itself to infinity again
    np.fill_lower(distance_matrix_copy, np.inf)     # Set the distance from a node to itself to infinity again
    
    np.fill_diagonal(distance_matrix_copy, 0)       # Set the distance from a node to itself to 0
    
    np.all(np.isfinite(distance_matrix_copy), axis=0, out=distance_matrix_copy)  # Check for negative cycles and set them to infinity
    np.all(np.isfinite(distance_matrix_copy), axis=1, out=distance_matrix_copy)  # Check for negative cycles and set them to infinity
    
    # The distance_matrix_copy now contains the shortest paths
    return distance_matrix_copy