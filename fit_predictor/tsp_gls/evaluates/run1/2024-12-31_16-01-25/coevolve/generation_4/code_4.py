import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal elements are zero
    # Calculate the minimum distance from each node to all other nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Create a matrix where each element is the difference between the corresponding element in the distance matrix
    # and the minimum distance from the node that the edge originates.
    heuristics_matrix = distance_matrix - min_distances[:, np.newaxis]
    
    # The heuristics matrix now contains values indicating how bad it is to include each edge in a solution
    return heuristics_matrix