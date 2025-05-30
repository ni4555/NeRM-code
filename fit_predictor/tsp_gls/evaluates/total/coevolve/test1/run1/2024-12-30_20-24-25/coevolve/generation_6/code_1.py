import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance is used for the heuristic matrix
    # The Manhattan distance heuristic for each edge is the sum of the distances
    # of the rows and columns of the edge, subtracted by the total number of rows/columns
    # to normalize the values between 0 and 1.
    num_nodes = distance_matrix.shape[0]
    heuristic_matrix = np.sum(distance_matrix, axis=0) + np.sum(distance_matrix, axis=1)
    return heuristic_matrix / (2 * num_nodes - 2)