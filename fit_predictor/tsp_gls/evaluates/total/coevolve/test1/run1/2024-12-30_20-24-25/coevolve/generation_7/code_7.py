import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate Chebyshev distance for each edge
    chebyshev_distances = np.max(np.abs(distance_matrix - distance_matrix.T), axis=0)
    
    # You can choose to use the Chebyshev distances directly or add a small constant
    # to avoid zero distances which might be problematic for the heuristic.
    # For example, add 1e-10 to the Chebyshev distances:
    chebyshev_distances += 1e-10
    
    # The problem description does not specify what the heuristic function should return.
    # Assuming that a lower value for an edge indicates a better heuristic, we could
    # return the Chebyshev distances directly as a negative value (or vice versa).
    # Let's return the negative Chebyshev distances as an example:
    return -chebyshev_distances