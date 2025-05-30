import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance from each node to all other nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Calculate the heuristic value for each edge as the difference between the
    # minimum distance to the destination node and the current edge distance
    heuristics = np.array([min_distances - distance_matrix[i, j]
                           for i in range(distance_matrix.shape[0])
                           for j in range(distance_matrix.shape[1] if i != j else 0)])
    
    # Reshape the heuristics array to match the shape of the distance matrix
    heuristics = heuristics.reshape(distance_matrix.shape)
    
    # Replace negative heuristics with zeros, as they are not meaningful in this context
    heuristics[heuristics < 0] = 0
    
    return heuristics