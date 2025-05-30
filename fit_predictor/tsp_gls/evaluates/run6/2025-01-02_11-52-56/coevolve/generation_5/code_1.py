import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the simple distance-based heuristic (sum of distances for each edge)
    simple_heuristic = np.sum(distance_matrix, axis=1) + np.sum(distance_matrix, axis=0)
    
    # Calculate the symmetric distance matrix
    symmetric_distance_matrix = (np.sum(distance_matrix, axis=1) + 
                                 np.sum(distance_matrix, axis=0) - 
                                 np.diagonal(distance_matrix))
    
    # Combine heuristics for the final heuristic scores
    combined_heuristic = simple_heuristic - symmetric_distance_matrix
    
    return combined_heuristic