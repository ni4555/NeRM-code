import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the higher the distance, the "worse" it is to include the edge.
    # Invert the distances to use a heuristic where lower values are better.
    return 1 / (distance_matrix + 1e-8)  # Adding a small constant to avoid division by zero.

# Example usage:
# distance_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
# print(heuristics_v2(distance_matrix))