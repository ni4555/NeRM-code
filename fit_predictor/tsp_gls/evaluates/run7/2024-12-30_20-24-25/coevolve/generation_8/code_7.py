import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic matrix is calculated by taking the reciprocal of the distances
    # which is a common heuristic approach for TSP, where smaller distances are preferred.
    heuristic_matrix = 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
    return heuristic_matrix