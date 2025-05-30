import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and contains only positive values.
    # Calculate the reciprocal of the distance for each edge as the heuristic value.
    # This heuristic assumes that shorter distances are better, which is common for TSP.
    heuristic_values = 1.0 / (distance_matrix + 1e-8)  # Adding a small value to avoid division by zero.
    
    # The shape of the heuristic array should be the same as the distance matrix.
    assert heuristic_values.shape == distance_matrix.shape, "Heuristic values must have the same shape as the distance matrix."
    
    return heuristic_values