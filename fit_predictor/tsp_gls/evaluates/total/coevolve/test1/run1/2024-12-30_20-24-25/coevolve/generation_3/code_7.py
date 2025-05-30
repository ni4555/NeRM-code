import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The implementation of this function will depend on the specifics of the heuristic
    # used for the TSP problem. Here, I will provide a simple example of a heuristic
    # that computes the sum of the minimum distances to the nearest neighbor for each vertex.
    # This is a basic heuristic and should be replaced with a more sophisticated one
    # as needed for the hybrid evolutionary algorithm.
    
    # Get the number of vertices
    num_vertices = distance_matrix.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic values
    for i in range(num_vertices):
        # Find the minimum distance to a neighboring vertex for each vertex i
        min_distance = np.min(distance_matrix[i, :])
        # Assign this minimum distance to the corresponding entry in the heuristic matrix
        heuristic_matrix[i, :] = min_distance
        # Assign a large number to the diagonal to avoid considering the same vertex
        heuristic_matrix[i, i] = np.inf
    
    return heuristic_matrix