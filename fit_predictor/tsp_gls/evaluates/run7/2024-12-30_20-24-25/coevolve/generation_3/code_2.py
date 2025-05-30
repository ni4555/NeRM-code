import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric
    # The heuristic for each edge can be a simple negative of the distance
    # to ensure the lower the heuristic, the more preferable the edge.
    return -distance_matrix