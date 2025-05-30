import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This is a simple heuristic where we assume the shortest path between any two nodes is always the direct distance
    # between them. We'll use the reciprocal of the distance as a heuristic to discourage long edges and encourage short ones.
    # For the edge (i, j), the heuristic is 1/dist(i, j), with a large constant added to the denominator to ensure non-zero
    # heuristic values.
    
    # Determine the maximum distance in the matrix to add a large constant
    max_distance = np.max(distance_matrix)
    
    # Create the heuristic matrix where each element is the reciprocal of the distance with a constant added
    heuristic_matrix = 1 / (distance_matrix + max_distance)
    
    # If the distance is zero, we want to discourage this edge from being included, so we can add a very large
    # number to these values to effectively disable them in the heuristic.
    # This is typically the case for the edges connecting a node to itself.
    self_loops = np.diag_indices_from(distance_matrix)
    heuristic_matrix[self_loops] = 1 / (max_distance * 10)  # Adding a large number to self-loop distances
    
    return heuristic_matrix

# Example usage:
# Create a small distance matrix
distance_matrix = np.array([[0, 2, 9, 10],
                            [1, 0, 6, 4],
                            [15, 7, 0, 8],
                            [6, 3, 12, 0]])

# Call the heuristic function
heuristic_matrix = heuristics_v2(distance_matrix)

print(heuristic_matrix)