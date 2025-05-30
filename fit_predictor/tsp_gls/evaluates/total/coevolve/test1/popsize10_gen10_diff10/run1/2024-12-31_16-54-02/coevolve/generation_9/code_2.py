import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    # The heuristic will be the negative of the distance matrix for the purpose of minimization
    return -distance_matrix