import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal elements are zero
    # The heuristic could be the negative of the distance (since smaller distances are better)
    # to reflect the "badness" of including an edge.
    # However, since the problem description mentions a "precision heuristic matrix",
    # we will create a more complex heuristic that includes some sort of precision factor.
    
    # Calculate the negative of the distances to create a heuristic that favors shorter paths
    heuristic_matrix = -distance_matrix
    
    # To simulate precision, we can add a small noise to the diagonal to avoid self-loops
    # which are not an issue but could potentially influence the heuristic matrix.
    # The noise is added only to the diagonal elements.
    precision_factor = 0.001
    np.fill_diagonal(heuristic_matrix, heuristic_matrix.diagonal() + precision_factor)
    
    return heuristic_matrix