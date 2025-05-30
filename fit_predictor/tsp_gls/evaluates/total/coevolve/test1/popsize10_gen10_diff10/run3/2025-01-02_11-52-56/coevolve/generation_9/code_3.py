import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This is a simple heuristic that returns the negative of the distance matrix
    # since we want to maximize the fitness, and shorter paths are better.
    return -distance_matrix

# Example usage:
# distance_matrix = np.array([[0, 2, 9], [1, 0, 10], [15, 8, 0]])
# heuristics = heuristics_v2(distance_matrix)
# print(heuristics)