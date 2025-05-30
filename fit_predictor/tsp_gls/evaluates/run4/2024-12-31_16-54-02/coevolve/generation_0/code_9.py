import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # In this example heuristic, we use the negative of the distance matrix as a simple heuristic
    # where a lower value means a "better" edge.
    return -distance_matrix