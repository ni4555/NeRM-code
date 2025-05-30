import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # The heuristic here is a simple one: the negative of the distance, as shorter edges
    # should be preferred in a TSP context.
    return -distance_matrix