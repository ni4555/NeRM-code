import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is a square matrix where the diagonal elements are 0
    # and the off-diagonal elements represent the distance between the respective cities.
    # We will create a heuristic matrix that penalizes shorter distances, assuming
    # that we want to avoid shorter paths as much as possible, which can be counterintuitive
    # but might work well with the algorithm's design.
    
    # The heuristic matrix is the negative of the distance matrix for simplicity.
    # We could also add some constant to ensure the matrix is positive if the distance
    # matrix has zero elements (which is not a good practice in this context).
    
    # However, since we are dealing with distances, we assume all elements are non-zero.
    # Hence, we can use the negative of the distance matrix directly.
    
    return -distance_matrix