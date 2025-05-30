import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is symmetric
    if not np.array_equal(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric.")

    # Calculate Chebyshev distance for each edge
    chebyshev_distances = np.maximum(distance_matrix, np.maximum(distance_matrix.T, distance_matrix))

    # Calculate a simple heuristic by combining Chebyshev distance with direct distance
    # The exact balance between the two can be tuned for better performance
    balance_factor = 0.5  # This can be adjusted for different scenarios
    heuristic_values = balance_factor * chebyshev_distances + (1 - balance_factor) * distance_matrix

    return heuristic_values