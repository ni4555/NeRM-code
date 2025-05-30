import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Using Manhattan distance heuristic as described
    # The Manhattan distance between two points (x1, y1) and (x2, y2) is given by:
    # Manhattan_distance = |x1 - x2| + |y1 - y2|
    # Since we are dealing with a distance matrix, we don't have coordinates,
    # but we can assume that the matrix has the same dimensions and use the row and column indices as x and y.
    # We will calculate the Manhattan distance for all pairs of rows and columns.
    
    # The shape of the distance matrix is (n, n), where n is the number of nodes.
    n = distance_matrix.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute Manhattan distance for each edge
    for i in range(n):
        for j in range(i + 1, n):  # avoid duplicate edges
            # Calculate Manhattan distance by summing the absolute differences of corresponding elements
            manhattan_distance = np.abs(distance_matrix[i] - distance_matrix[j]).sum()
            heuristics_matrix[i, j] = heuristics_matrix[j, i] = manhattan_distance
            
    return heuristics_matrix