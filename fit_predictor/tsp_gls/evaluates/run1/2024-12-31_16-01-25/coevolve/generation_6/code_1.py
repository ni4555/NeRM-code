import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance heuristic for this example
    Manhattan_distance = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    # Normalize by the maximum possible Manhattan distance to get a heuristic value
    max_manhattan_distance = Manhattan_distance.max()
    heuristic_values = Manhattan_distance / max_manhattan_distance
    return heuristic_values