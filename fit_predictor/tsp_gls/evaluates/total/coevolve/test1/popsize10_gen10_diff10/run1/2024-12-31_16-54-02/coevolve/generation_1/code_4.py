import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and contains zeros on the diagonal
    # The heuristic for an edge from node i to node j is the negative of the distance
    # because we are minimizing the total distance
    return -distance_matrix