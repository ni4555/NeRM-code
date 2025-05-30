import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Using the distance matrix as the heuristic matrix. This is a straightforward
    # approach where the heuristic for each edge is the actual distance between
    # those two cities.
    return distance_matrix