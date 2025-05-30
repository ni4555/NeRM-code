import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Example heuristic: Calculate the sum of distances from each node to the origin node
    origin_node = 0  # Assuming the origin node is the first node
    heuristic_matrix = np.sum(distance_matrix, axis=1) - distance_matrix[:, origin_node]
    return heuristic_matrix