import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the "badness" of each edge as the negative of the distance
    # We use negative because some algorithms may prefer minimizing a cost function.
    badness_matrix = -distance_matrix
    return badness_matrix