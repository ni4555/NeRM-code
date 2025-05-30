import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the "badness" of an edge is the inverse of its distance
    # If distance is zero (invalid edge), we assign a high "badness" value
    badness_matrix = np.reciprocal(distance_matrix)
    badness_matrix[np.isinf(badness_matrix)] = np.inf  # Replace -inf with inf
    badness_matrix[np.isnan(badness_matrix)] = np.inf  # Replace NaN with inf
    return badness_matrix