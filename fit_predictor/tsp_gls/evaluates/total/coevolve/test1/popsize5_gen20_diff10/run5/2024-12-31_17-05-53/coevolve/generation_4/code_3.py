import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Manhattan distance for heuristic guidance
    manhattan_distance = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    
    # Calculate average distance for edge selection
    average_distance = np.mean(distance_matrix, axis=0)
    
    # Combine the Manhattan distance and average distance
    # Here, we could use a weighted sum or another combination method
    # For simplicity, we'll use a linear combination where both are equally weighted
    heuristics = 0.5 * manhattan_distance + 0.5 * average_distance
    
    # Ensure that the heuristics are non-negative
    heuristics = np.maximum(0, heuristics)
    
    return heuristics