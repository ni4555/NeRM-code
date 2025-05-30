import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Subtract the minimum distance from each row to normalize distances
    min_distance_per_row = np.min(distance_matrix, axis=1, keepdims=True)
    normalized_distance_matrix = distance_matrix - min_distance_per_row

    # Heuristic: Calculate the sum of each row as an indicator of the desirability
    # This sum represents the total additional distance over the minimum distance
    heuristic_values = np.sum(normalized_distance_matrix, axis=1)

    return heuristic_values