import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance between each pair of nodes
    manhattan_distance_matrix = np.abs(distance_matrix - distance_matrix.T)
    
    # The heuristic value for each edge can be seen as the minimum Manhattan distance
    # between the two nodes, assuming a direct connection.
    # This is a simplification and might not reflect the actual TSP cost, but serves
    # as a heuristic estimate.
    heuristic_matrix = np.min(manhattan_distance_matrix, axis=1)
    
    return heuristic_matrix