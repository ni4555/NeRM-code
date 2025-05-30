import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the output matrix with the same shape as the input matrix
    # where each entry (i, j) will represent the badness of including edge (i, j)
    badness_matrix = np.full_like(distance_matrix, np.inf)
    
    # Calculate the diagonal elements as 0 since the distance to the node itself
    # is not considered bad
    np.fill_diagonal(badness_matrix, 0)
    
    # Calculate the shortest path between any two nodes using a heuristic approach
    # and populate the badness matrix accordingly
    # We can use a simple heuristic like the maximum distance from the node to any other node
    # as a proxy for badness.
    for i in range(distance_matrix.shape[0]):
        # Find the maximum distance from node i to any other node
        max_distance = np.max(distance_matrix[i, :])
        # Update the badness matrix with this value for all edges connected to node i
        badness_matrix[i, :] = max_distance
    
    # Normalize the badness matrix by dividing each row by the sum of its values
    # This ensures that the sum of badness values for each node is equal to 1
    row_sums = np.sum(badness_matrix, axis=1, keepdims=True)
    badness_matrix = badness_matrix / row_sums
    
    return badness_matrix

# Example usage:
# distance_matrix = np.array([[0, 2, 9], [1, 0, 10], [15, 6, 0]])
# print(heuristics_v2(distance_matrix))