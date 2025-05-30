import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    min_distance = np.min(distance_matrix)
    total_cost = np.sum(distance_matrix)
    # Subtract the minimum distance and normalize by the total cost
    heuristics = (distance_matrix - min_distance) / total_cost
    return heuristics