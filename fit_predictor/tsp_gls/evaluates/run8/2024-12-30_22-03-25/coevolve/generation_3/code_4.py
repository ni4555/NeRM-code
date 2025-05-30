import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    # The heuristic here is a simple inverse of the distance (smaller distance is better)
    # The idea is that the heuristic should be larger for shorter distances, which suggests that
    # including this edge might be more beneficial for the TSP tour.
    heuristic_matrix = 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    
    return heuristic_matrix