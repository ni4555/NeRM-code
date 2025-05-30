import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the maximum distance in the matrix to initialize the edge cost matrix
    max_distance = np.max(distance_matrix)
    
    # Initialize the heuristic matrix with high values (considering high cost)
    heuristics_matrix = np.full(distance_matrix.shape, max_distance)
    
    # Calculate the initial heuristic for each edge based on a dynamic shortest path algorithm
    # For simplicity, we'll just use the negative of the distance as the heuristic (since shorter distances are better)
    # Note: In a real-world scenario, the dynamic shortest path algorithm would be more complex and could be used to calculate
    # the shortest path to all other nodes from each node, which would be used to derive a more sophisticated heuristic.
    heuristics_matrix = -distance_matrix
    
    # To prevent node repetition and guarantee a seamless route traversal, set the heuristic for edges leading back to the origin to 0
    # (assuming the origin node is at index 0)
    heuristics_matrix[:, 0] = 0
    heuristics_matrix[0, :] = 0
    
    # Return the heuristic matrix which is the same shape as the input distance matrix
    return heuristics_matrix