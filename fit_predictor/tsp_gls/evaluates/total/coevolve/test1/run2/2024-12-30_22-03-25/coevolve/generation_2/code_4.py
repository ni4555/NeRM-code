import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the average distance between each pair of nodes
    average_distances = np.mean(distance_matrix, axis=0)
    
    # Calculate the standard deviation of distances for each node
    std_distances = np.std(distance_matrix, axis=0)
    
    # Create a heuristic value for each edge based on the average and standard deviation
    # We use a heuristic that combines the average distance and the standard deviation
    # This heuristic assumes that edges with higher average distance and higher standard deviation
    # are more likely to be included in the solution
    heuristic_values = (average_distances + std_distances) ** 2
    
    return heuristic_values