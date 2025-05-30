import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the average distance in the matrix
    average_distance = np.mean(distance_matrix)
    
    # Create a boolean matrix where True indicates shorter than average edges
    is_shorter = distance_matrix < average_distance
    
    # Return a matrix of the same shape with True for shorter edges
    return is_shorter.astype(int)