import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This function should implement a heuristic to estimate the cost of each edge.
    # Since the problem description does not provide specific details about the heuristic,
    # we will assume a simple heuristic: the lower the distance, the better the edge.
    # We will use the reciprocal of the distance to approximate the "badness" of an edge.
    # This means edges with shorter distances will have lower heuristic values.
    
    # To avoid division by zero, we add a small constant to the reciprocal.
    epsilon = 1e-8
    return 1 / (distance_matrix + epsilon)