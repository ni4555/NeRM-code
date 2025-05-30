import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    
    # Calculate Manhattan distances between each pair of nodes
    manhattan_distances = np.abs(distance_matrix - np.diag(np.diag(distance_matrix)))
    
    # The Manhattan distance is the sum of the absolute differences
    # We want to return a measure of "badness", so we'll use the sum of the distances
    # instead of the actual Manhattan distance, which could be positive or zero.
    badness_measure = np.sum(manhattan_distances, axis=1)
    
    return badness_measure