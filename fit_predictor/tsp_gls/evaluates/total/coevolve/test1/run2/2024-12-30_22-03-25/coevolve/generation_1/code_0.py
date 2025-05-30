import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the inverse of the distance matrix to get a heuristic
    # that suggests including shorter edges first.
    # This is a very basic heuristic and might not be suitable for all scenarios.
    heuristics = 1.0 / distance_matrix
    # Replace any NaN values with a large number (e.g., np.inf) to handle zero distances
    # which would otherwise result in an infinite heuristic value.
    np.nan_to_num(heuristics, nan=np.inf, copy=False)
    return heuristics