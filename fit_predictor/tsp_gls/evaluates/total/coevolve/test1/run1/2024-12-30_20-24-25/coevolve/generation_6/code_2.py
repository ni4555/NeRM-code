import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Compute Manhattan distance heuristics for each edge
    # Manhattan distance is the sum of the absolute differences of their Cartesian coordinates
    # We use Manhattan distance on the indices of the matrix to represent edge costs
    heuristics = np.abs(np.subtract.outer(np.arange(distance_matrix.shape[0]), np.arange(distance_matrix.shape[0]))).sum(axis=1)
    return heuristics