import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the average distance for each edge
    edge_average_distances = np.mean(distance_matrix, axis=0)
    
    # Set the heuristic value for each edge as the average distance
    # This will be used to guide the edge selection process
    heuristic_matrix = edge_average_distances
    
    return heuristic_matrix