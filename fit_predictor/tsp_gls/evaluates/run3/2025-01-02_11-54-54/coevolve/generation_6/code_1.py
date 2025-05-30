import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an empty array to hold the heuristics
    heuristics = np.zeros_like(distance_matrix, dtype=np.float64)
    
    # Get the shape of the distance matrix
    rows, cols = distance_matrix.shape
    
    # Iterate over each row in the distance matrix
    for i in range(rows):
        # Calculate the minimum and maximum distance for the current row
        min_distance = np.min(distance_matrix[i])
        max_distance = np.max(distance_matrix[i])
        
        # Calculate the heuristic for the current row
        heuristics[i] = max_distance - min_distance
    
    return heuristics