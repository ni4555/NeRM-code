import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function is a simple heuristic that assumes lower distance values
    # indicate a better edge to include in a solution. This can be replaced
    # with more complex heuristics depending on the problem's requirements.

    # Calculate the maximum distance in the matrix to normalize the values
    max_distance = np.max(distance_matrix)
    
    # Normalize the distance matrix to have values between 0 and 1
    normalized_matrix = distance_matrix / max_distance
    
    # Return the normalized matrix which acts as a heuristic indicator
    return normalized_matrix