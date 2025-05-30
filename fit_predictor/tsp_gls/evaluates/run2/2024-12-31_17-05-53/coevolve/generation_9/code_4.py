import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Manhattan distances between all pairs of cities
    manhattan_distances = np.abs(distance_matrix - np.roll(distance_matrix, 1, axis=0)) + \
                          np.abs(distance_matrix - np.roll(distance_matrix, 1, axis=1))
    
    # Calculate the average Manhattan distance
    average_distance = np.mean(manhattan_distances)
    
    # Create the heuristic matrix
    heuristic_matrix = np.where(distance_matrix > 0, average_distance - distance_matrix, 0)
    
    return heuristic_matrix