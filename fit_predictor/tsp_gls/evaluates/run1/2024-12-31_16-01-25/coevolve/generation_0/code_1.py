import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # since there is no cost to stay at the same node.
    # Return the distance matrix itself as a heuristic.
    return distance_matrix.copy()