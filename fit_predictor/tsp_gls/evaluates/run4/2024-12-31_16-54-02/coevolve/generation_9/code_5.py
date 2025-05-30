import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The "badness" of an edge is inversely proportional to its distance.
    # The smaller the distance, the lower the "badness".
    # We can represent "badness" as the inverse of the distance, with a small epsilon
    # to avoid division by zero when the distance is zero.
    epsilon = 1e-6
    return 1.0 / (distance_matrix + epsilon)