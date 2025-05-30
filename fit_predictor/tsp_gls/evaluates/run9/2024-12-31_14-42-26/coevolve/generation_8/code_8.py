import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Adding a small constant to the inverse of the distances to avoid division by zero
    min_distance = np.min(distance_matrix)
    if min_distance == 0:
        min_distance = 1e-6  # Replace with a small positive number if distances are guaranteed to be positive

    # Calculate the heuristic based on the inverse of the distances
    # and add a constant to ensure all values are positive and finite
    heuristics = (1 / (distance_matrix + min_distance))

    return heuristics