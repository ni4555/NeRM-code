import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assume that the distance matrix is symmetric, i.e., d[i][j] == d[j][i]
    # We will use a heuristic based on a weighted sum of the Chebyshev and Euclidean distances
    
    # Calculate Chebyshev distances
    chebyshev_matrix = np.maximum.reduce(distance_matrix, axis=0)
    
    # Calculate Euclidean distances
    # We need to handle the diagonal values (which are 0) by adding the square root of the sum of squares of the other dimensions
    euclidean_matrix = np.array([[np.sqrt(np.sum(np.square(distance_matrix[i]))) if i != j else 0 for j in range(len(distance_matrix))] for i in range(len(distance_matrix))])
    
    # Define weights for the heuristic, these can be tuned for performance
    chebyshev_weight = 0.5
    euclidean_weight = 0.5
    
    # Combine Chebyshev and Euclidean distances with the weights
    combined_distances = chebyshev_weight * chebyshev_matrix + euclidean_weight * euclidean_matrix
    
    # The heuristic function is a measure of how "bad" it is to include each edge in a solution
    # Here, we will simply negate the combined distances to get the prior indicators
    # (the lower the value, the better the edge)
    heuristic_matrix = -combined_distances
    
    return heuristic_matrix