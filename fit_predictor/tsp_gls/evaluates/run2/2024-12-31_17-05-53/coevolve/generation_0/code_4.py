import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with the same shape as the distance_matrix
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics based on some heuristic function
    # For example, we can use the Manhattan distance as a heuristic
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Calculate Manhattan distance as heuristic value
                heuristics_matrix[i][j] = abs(i - j)
    
    return heuristics_matrix