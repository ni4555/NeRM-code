import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is based on the minimum distance to any node from the current node
    # For simplicity, let's use the minimum distance to the first node in each row as the heuristic value
    min_distances = np.min(distance_matrix, axis=1)
    return min_distances