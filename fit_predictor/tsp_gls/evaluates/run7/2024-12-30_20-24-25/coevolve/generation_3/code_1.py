import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assume the distance_matrix is square and symmetric
    num_vertices = distance_matrix.shape[0]
    
    # Create a matrix to store the heuristic values
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # The origin is at (0,0) in the coordinate system
    origin = np.zeros(num_vertices)
    
    # Calculate the Manhattan distance from the origin to each vertex
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                heuristic_matrix[i][j] = np.abs(i - origin[0]) + np.abs(j - origin[1])
    
    return heuristic_matrix