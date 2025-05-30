import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance between all pairs of cities as a simple heuristic
    heuristics = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    
    # Apply a slight perturbation to encourage diversity
    np.random.shuffle(heuristics)
    
    return heuristics