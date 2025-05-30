import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is a 2D array with dimensions (n, n) where n is the number of nodes.
    # We'll calculate the Manhattan distance for each edge and use that as the heuristic value.
    # The Manhattan distance between two points (x1, y1) and (x2, y2) is |x2 - x1| + |y2 - y1|.
    
    # For the sake of the heuristic, let's assume the distance matrix contains coordinates in some 2D space
    # and that the nodes are indexed by row and column, where each row represents the coordinates (x, y) of a node.
    # This is an assumption, as the distance matrix could contain actual distances.
    
    # Extract coordinates from the distance matrix, assuming it's in the shape of (n, 2)
    if distance_matrix.shape[1] != 2:
        raise ValueError("The distance matrix should have shape (n, 2) for (x, y) coordinates.")
    
    x_coords = distance_matrix[:, 0]
    y_coords = distance_matrix[:, 1]
    
    # Calculate Manhattan distance for each edge
    # The heuristic for each edge (i, j) is the Manhattan distance between node i and node j
    heuristic_matrix = np.abs(np.diff(x_coords, axis=0)) + np.abs(np.diff(y_coords, axis=0))
    
    # The heuristic values should be the same for each edge, but we can't simply take the first element
    # because the distance matrix might have negative values or not represent distances directly.
    # Therefore, we use the average Manhattan distance between any two consecutive nodes as the heuristic.
    # This is a simplification and may not be the exact heuristic described in the problem description.
    return heuristic_matrix.mean(axis=0)

# Example usage:
# distance_matrix = np.array([[0, 0], [1, 2], [3, 3], [0, 4]])
# heuristics = heuristics_v2(distance_matrix)
# print(heuristics)