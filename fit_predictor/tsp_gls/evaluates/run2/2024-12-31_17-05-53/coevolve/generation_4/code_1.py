import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # The Manhattan distance is used to estimate the cost of each edge
    # For a given edge from city i to city j, we calculate the Manhattan distance
    # which is the sum of the absolute differences of their Cartesian coordinates.
    # Here we are using the indices of the cities as the coordinates for simplicity.
    edge_cost = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    
    # The heuristic for each edge is inversely proportional to its cost.
    # This encourages the algorithm to prefer edges with lower cost.
    heuristics = 1 / edge_cost
    
    # Handle the case where the edge cost is zero (which would cause division by zero)
    # in such cases, we set the heuristic to a very high value to discourage the edge.
    heuristics[np.isclose(edge_cost, 0)] = np.inf
    
    return heuristics