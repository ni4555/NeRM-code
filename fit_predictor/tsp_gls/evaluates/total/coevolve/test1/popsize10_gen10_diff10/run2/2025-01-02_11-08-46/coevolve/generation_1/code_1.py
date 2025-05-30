import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This heuristic function computes the Chebyshev distance for each edge
    # which is the maximum of the absolute differences of corresponding coordinates.
    # Chebyshev distance is used as an example heuristic; others could be chosen based on the problem context.
    # The idea here is to create a heuristic that is more difficult for long edges, 
    # potentially guiding the metaheuristic to favor shorter paths.
    
    # Find the maximum distance for each edge to create a "heuristic" value
    max_distance = np.max(distance_matrix, axis=0)
    max_distance = np.max(distance_matrix, axis=1)  # Get the maximum of the transposed matrix
    
    # The Chebyshev heuristic is simply the maximum distance from the origin to a point
    # where the origin is the first city and the points are the other cities.
    # This is not a perfect heuristic for the TSP since it doesn't consider the total
    # distance or the order of the cities but serves as an example.
    heuristic_values = max_distance
    
    return heuristic_values