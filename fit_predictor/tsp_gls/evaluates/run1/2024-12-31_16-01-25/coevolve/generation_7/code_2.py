import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is zero
    # We use the Euclidean distance heuristic to estimate the "badness" of including each edge
    # The heuristic is simply the negative of the distance, assuming a smaller distance is better
    
    # The shape of the heuristics matrix will be the same as the distance matrix
    heuristics_matrix = -distance_matrix
    
    # We also need to handle the diagonal elements, which are zero in a distance matrix
    # Since we are using the negative distance as a heuristic, zero becomes "infinite" bad
    np.fill_diagonal(heuristics_matrix, np.inf)
    
    return heuristics_matrix