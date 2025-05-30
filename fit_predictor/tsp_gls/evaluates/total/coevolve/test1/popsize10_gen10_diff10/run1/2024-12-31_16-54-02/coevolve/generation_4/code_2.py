import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Fill the heuristics array with the negative of the distances
    # Negative distances are used because min-heap can be used in the priority queue
    heuristics[distance_matrix > 0] = -distance_matrix[distance_matrix > 0]
    
    return heuristics