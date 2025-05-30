import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that we want to penalize longer distances, we could simply use the negative
    # of the distance matrix as a heuristic. The shape of the returned array will be the same
    # as the input distance matrix.
    return -distance_matrix