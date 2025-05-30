import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Ensure the distance matrix is valid
    if not isinstance(distance_matrix, np.ndarray) or not np.issubdtype(distance_matrix.dtype, np.number):
        raise ValueError("The distance matrix must be a numpy array with numeric values.")
    
    # Create a matrix where each element is the reciprocal of the distance
    # Note: This implementation assumes all distances are positive and non-zero.
    # If there are zero distances, they will be considered as "infinite" badness.
    heuristic_matrix = 1 / distance_matrix
    
    # Replace any zeros with infinity to indicate that these edges should not be included.
    # This is a common approach in TSP heuristics to represent non-existent edges.
    # However, if the distance matrix is guaranteed to be non-zero, this step can be omitted.
    np.nan_to_num(heuristic_matrix, nan=np.inf, copy=False)
    
    return heuristic_matrix