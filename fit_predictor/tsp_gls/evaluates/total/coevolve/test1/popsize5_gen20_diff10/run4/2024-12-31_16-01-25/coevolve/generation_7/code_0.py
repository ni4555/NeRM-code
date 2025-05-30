import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the negative of the distance matrix
    # This assumes that smaller distances are preferable
    # You can adjust this logic based on your needs
    negative_distance_matrix = -distance_matrix
    return negative_distance_matrix

# Example usage:
# Assuming we have a 4x4 distance matrix
distance_matrix = np.array([
    [0, 5, 7, 8],
    [6, 0, 2, 6],
    [3, 3, 0, 9],
    [2, 8, 1, 0]
])

heuristic_matrix = heuristics_v2(distance_matrix)
print(heuristic_matrix)