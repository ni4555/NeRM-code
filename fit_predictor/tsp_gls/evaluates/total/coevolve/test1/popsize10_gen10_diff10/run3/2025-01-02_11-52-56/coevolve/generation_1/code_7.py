import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric (i.e., the distance from A to B is the same as from B to A)
    # and the distance from a city to itself is 0.
    # We calculate the sum of distances for all possible paths (excluding the trivial single-node paths).
    # We then return the inverse of these sums as our heuristic values. This is because a higher sum of
    # distances indicates a less efficient path, so we want to penalize paths with higher sums.
    
    # The shape of the distance matrix is (n, n), where n is the number of cities.
    n = distance_matrix.shape[0]
    
    # Initialize an array to hold our heuristic values, with the same shape as the distance matrix.
    # Start with all values set to zero.
    heuristic_values = np.zeros_like(distance_matrix)
    
    # Loop through each pair of cities (i, j) where i != j.
    for i in range(n):
        for j in range(i + 1, n):  # j is exclusive of n because we already calculated i
            # Sum the distances for the path i-j (i to j)
            path_sum = np.sum(distance_matrix[i, :]) + distance_matrix[i, j]
            
            # Calculate the heuristic value by taking the inverse of the sum of distances
            # We add a small constant to prevent division by zero.
            heuristic_values[i, j] = 1 / (path_sum + 1e-10)
    
    # Since the distance matrix is symmetric, we set the values for j-i to be the same as for i-j.
    # This is because the distance from i to j is the same as the distance from j to i.
    heuristic_values = (heuristic_values + heuristic_values.T) / 2
    
    return heuristic_values