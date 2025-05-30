import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the average distance of all edges
    average_distance = np.mean(distance_matrix)
    
    # Compute the heuristics for each edge
    heuristics = np.where(distance_matrix > average_distance, distance_matrix, 0)
    
    return heuristics