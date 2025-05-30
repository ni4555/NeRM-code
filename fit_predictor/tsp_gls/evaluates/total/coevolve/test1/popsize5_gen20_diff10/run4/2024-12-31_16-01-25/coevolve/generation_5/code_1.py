import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic should be the inverse of the distance, 
    # as shorter paths are preferable. This is a common heuristic for the TSP.
    return 1.0 / distance_matrix