import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function calculates a heuristic value for each edge in the distance matrix.
    # For simplicity, let's assume the heuristic is the reciprocal of the distance (since we want to prioritize shorter distances).
    # This is a naive heuristic that assumes the shorter the distance, the better the edge.
    # Note: This is a placeholder for a more sophisticated heuristic that would be implemented as per the problem description.
    return 1.0 / (distance_matrix + 1e-10)  # Adding a small value to avoid division by zero.