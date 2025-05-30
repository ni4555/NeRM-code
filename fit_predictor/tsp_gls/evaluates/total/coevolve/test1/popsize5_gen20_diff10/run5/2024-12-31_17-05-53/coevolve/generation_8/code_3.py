import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the heuristic is a simple inverse of the distance for the sake of this example.
    # In practice, this should be a more sophisticated heuristic based on the problem domain.
    return 1 / (distance_matrix + 1e-10)  # Adding a small epsilon to avoid division by zero.