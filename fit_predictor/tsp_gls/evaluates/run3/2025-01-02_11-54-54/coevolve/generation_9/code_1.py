import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    min_distance = np.min(distance_matrix)
    total_cost = np.sum(distance_matrix)
    normalized_distances = distance_matrix - min_distance
    correlation_with_total_cost = normalized_distances / total_cost
    return correlation_with_total_cost