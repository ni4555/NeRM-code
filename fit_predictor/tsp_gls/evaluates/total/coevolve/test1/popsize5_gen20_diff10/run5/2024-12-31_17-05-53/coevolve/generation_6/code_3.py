import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Manhattan distance
    manhattan_distance = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    
    # Calculate average distance
    average_distance = np.mean(distance_matrix, axis=1)
    
    # Combine Manhattan distance and average distance
    # We could use a simple linear combination, but here we are using a more complex
    # formula that gives more weight to the Manhattan distance.
    # The coefficients are arbitrary and could be adjusted for different scenarios.
    heuristics = 1.5 * manhattan_distance + 0.5 * average_distance
    
    # Normalize the heuristics to make them comparable
    max_heuristic = np.max(heuristics)
    heuristics = heuristics / max_heuristic
    
    return heuristics