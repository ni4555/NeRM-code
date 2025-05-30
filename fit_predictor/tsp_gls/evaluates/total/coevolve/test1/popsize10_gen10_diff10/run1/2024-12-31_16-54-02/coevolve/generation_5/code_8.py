import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance from each node to all other nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Calculate the heuristic value for each edge as the difference
    # between the total distance of the cycle (including the return to the origin)
    # and the sum of the minimum distances of each edge.
    # Note: The sum of the minimum distances will be double-counted for the return to the origin,
    # so we subtract one instance of the origin's minimum distance.
    total_min_distance = np.sum(min_distances)
    heuristic_values = total_min_distance - min_distances - np.min(min_distances)
    
    # The heuristic values are negative because we are trying to minimize the total distance.
    # We return the absolute values as the input function signature expects a shape matching the input.
    return np.abs(heuristic_values)