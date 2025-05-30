import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Get the number of rows and columns from the distance matrix
    num_cities = distance_matrix.shape[0]
    
    # Calculate the center of the matrix
    center = num_cities // 2
    
    # Initialize an empty heuristic matrix with the same shape as the distance matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Iterate over each edge in the distance matrix
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:  # Skip the diagonal (self-loops)
                # Calculate the Manhattan distance from the center to the city at (i, j)
                heuristic = abs(center - i) + abs(center - j)
                # Assign the heuristic to the corresponding edge in the heuristic matrix
                heuristic_matrix[i, j] = heuristic
    
    return heuristic_matrix