import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The Manhattan distance heuristic for each edge
    # The score for an edge (i, j) is the sum of the absolute differences
    # between the i-th city's coordinates and the j-th city's coordinates.
    # Since we are using Manhattan distance, we only need the absolute differences
    # of the respective indices, as the distance matrix itself contains the distances.
    return np.abs(np.arange(distance_matrix.shape[0])[:, None] - np.arange(distance_matrix.shape[1]))

# Example usage:
# Create a sample distance matrix with 5 cities.
distance_matrix = np.array([
    [0, 3, 1, 4, 2],
    [3, 0, 3, 5, 4],
    [1, 3, 0, 2, 1],
    [4, 5, 2, 0, 3],
    [2, 4, 1, 3, 0]
])

# Get the heuristic scores for each edge.
heuristic_scores = heuristics_v2(distance_matrix)
print(heuristic_scores)