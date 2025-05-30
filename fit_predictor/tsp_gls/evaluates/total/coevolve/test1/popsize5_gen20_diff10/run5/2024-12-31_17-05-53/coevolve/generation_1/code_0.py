import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and the diagonal is filled with zeros
    # The heuristic is a simple negative of the distance matrix, since we want to minimize
    # the total distance in the TSP. The negative values will be used to indicate "goodness"
    # or "badness" of including each edge in the solution.
    return -distance_matrix