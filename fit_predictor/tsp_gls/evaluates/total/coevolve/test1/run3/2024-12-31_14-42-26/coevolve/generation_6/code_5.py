import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic values based on the distance matrix
    # For simplicity, we'll use the inverse of the distance as the heuristic value
    # This is a common approach where smaller distances are preferred
    heuristic_matrix = 1 / (distance_matrix + 1e-10)  # Adding a small value to avoid division by zero
    
    return heuristic_matrix