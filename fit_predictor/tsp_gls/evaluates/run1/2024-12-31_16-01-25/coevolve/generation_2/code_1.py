import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Invert the distance matrix so that smaller distances correspond to lower "badness" values
    # This is a simple heuristic where we think shorter distances are better to include in the TSP solution
    badness_matrix = 1 / (distance_matrix + 1e-8)  # Adding a small epsilon to avoid division by zero
    return badness_matrix

# Example usage:
# distance_matrix = np.array([[0, 2, 9, 10],
#                             [1, 0, 6, 4],
#                             [15, 7, 0, 8],
#                             [6, 3, 12, 0]])
# print(heuristics_v2(distance_matrix))