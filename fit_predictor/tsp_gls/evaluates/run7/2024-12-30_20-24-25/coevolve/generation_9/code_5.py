import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is symmetric
    assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix must be symmetric."

    # Use the inverse of the distance as a heuristic, assuming zero distance for diagonal elements
    # (self-loops are not allowed in the TSP)
    heuristic_matrix = 1.0 / (distance_matrix + 1e-10)  # Adding a small value to avoid division by zero
    heuristic_matrix[np.isinf(heuristic_matrix)] = 0.0  # Replace infinities with zeros
    return heuristic_matrix