import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The heuristics will be the inverse of the distance matrix, since we want
    # to minimize the total distance traveled. However, to ensure that the
    # heuristic values are non-negative, we will subtract the smallest value in
    # the distance matrix from all the elements in the distance matrix before
    # taking the inverse.
    min_distance = np.min(distance_matrix)
    adjusted_distance_matrix = distance_matrix - min_distance
    # Taking the inverse of the adjusted distance matrix as the heuristic.
    heuristics = 1.0 / adjusted_distance_matrix
    # Replace any NaNs (which occur due to division by zero) with a large negative value.
    heuristics[np.isnan(heuristics)] = -np.inf
    return heuristics