import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric (distance from i to j is the same as from j to i)
    # We'll use a simple heuristic: the heuristic value for an edge (i, j) is the negative of the distance
    # since we're looking for a minimum, and we want to encourage the inclusion of shorter edges.
    return -distance_matrix