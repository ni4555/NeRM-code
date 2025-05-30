import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the negative of the distance matrix to use as the heuristic
    # The negative values indicate the cost of not including the edge
    return -distance_matrix