import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming a simple heuristic where longer distances have higher penalty.
    # This could be adjusted based on more complex criteria.
    return 1 / (1 + distance_matrix)  # Using a simple inverse heuristic

# Example usage:
# distance_matrix = np.array([[0, 2, 9, 10],
#                             [1, 0, 6, 4],
#                             [15, 7, 0, 8],
#                             [6, 3, 12, 0]])
# print(heuristics_v2(distance_matrix))