import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic where the cost of each edge is inversely proportional to its distance
    # This heuristic assumes that shorter distances are preferable
    return 1.0 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero