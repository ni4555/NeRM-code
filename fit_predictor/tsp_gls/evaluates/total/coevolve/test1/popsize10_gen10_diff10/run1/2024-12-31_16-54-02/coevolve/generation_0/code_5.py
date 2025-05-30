import numpy as np
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Number of nodes in the distance matrix
    num_nodes = distance_matrix.shape[0]
    
    # Calculate the total direct distance (sum of all diagonal elements)
    total_direct_distance = np.sum(distance_matrix.diagonal())
    
    # Calculate the MST
    mst = minimum_spanning_tree(distance_matrix)
    
    # Get the total distance of the MST
    mst_total_distance = np.sum(mst.data)
    
    # Calculate the cost of not including each edge in the MST
    # This is done by subtracting the MST total distance from the total direct distance
    # The result is then divided by the total direct distance to normalize
    heuristic_values = (total_direct_distance - mst_total_distance) / total_direct_distance
    
    # The resulting array will have the same shape as the distance matrix
    # The diagonal elements (self-loops) are set to 0 since they are not "bad" edges
    heuristics = np.zeros_like(distance_matrix)
    heuristics[np.triu_indices_from(heuristics, k=1)] = heuristic_values
    
    return heuristics