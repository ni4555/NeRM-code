import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the total distance of the complete cycle (sum of all edges)
    total_distance = np.sum(distance_matrix)
    
    # Calculate the minimum distance between each pair of nodes
    min_distances = np.min(distance_matrix, axis=0)
    
    # Create a matrix where each entry is the difference between the total distance and the minimum distance
    # between the current node and the starting node (which is the same for all nodes)
    heuristics_matrix = total_distance - min_distances
    
    return heuristics_matrix