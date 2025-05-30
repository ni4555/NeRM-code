import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the diagonal (self-loops) and set them to a large value
    np.fill_diagonal(heuristic_matrix, np.inf)
    
    # Set the minimum distance between any two nodes as the heuristic
    np.minimum.reduceat(heuristic_matrix, range(heuristic_matrix.shape[0]), axis=1, out=heuristic_matrix)
    np.minimum.reduceat(heuristic_matrix, range(heuristic_matrix.shape[1]), axis=0, out=heuristic_matrix)
    
    # Return the heuristic matrix
    return heuristic_matrix