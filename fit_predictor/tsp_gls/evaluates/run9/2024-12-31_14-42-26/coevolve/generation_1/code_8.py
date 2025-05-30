import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is based on the average distance of each edge
    # This is a simple example heuristic and should be adapted to the specific needs of the problem
    return np.mean(distance_matrix, axis=0)