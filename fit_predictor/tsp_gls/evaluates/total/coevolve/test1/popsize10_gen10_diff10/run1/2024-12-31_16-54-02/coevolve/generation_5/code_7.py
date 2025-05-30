import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is simply the negative of the edge distances
    # for a given TSP instance represented by the distance_matrix.
    # This is a common approach where shorter edges have higher heuristic values.
    
    # The result should have the same shape as distance_matrix.
    return -distance_matrix