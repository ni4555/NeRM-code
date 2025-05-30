import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Chebyshev distance matrix as a proxy for the heuristic values
    # The Chebyshev distance between two points is the maximum absolute difference of their coordinates.
    # This is a simple heuristic that assumes the worst-case distance to the nearest neighbor as the cost to include an edge.
    Chebyshev_dist = np.max(np.abs(distance_matrix - np.min(distance_matrix, axis=0)), axis=0)
    
    # Since the Chebyshev distance could result in a very large heuristic for the first and last cities,
    # we need to adjust these to avoid infeasible solutions where a city is visited twice.
    # This can be done by setting the heuristic for the first and last city to the distance to the nearest city.
    Chebyshev_dist[0] = distance_matrix[0][1]  # Distance from first city to the second city
    Chebyshev_dist[-1] = distance_matrix[-1][-2]  # Distance from the last city to the second last city
    
    return Chebyshev_dist