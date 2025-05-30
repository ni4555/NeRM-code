import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Apply Manhattan distance heuristic to estimate edge inclusion cost
    # For simplicity, let's assume that the Manhattan distance is the sum of the absolute differences
    # in the coordinates of the cities. This is a simplification and might not be optimal for all cases.
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Calculate Manhattan distance between cities i and j
                manhattan_distance = np.sum(np.abs(distance_matrix[i] - distance_matrix[j]))
                # Assign the estimated cost to the corresponding position in the heuristics matrix
                heuristics_matrix[i][j] = manhattan_distance
    
    return heuristics_matrix