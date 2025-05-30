import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the minimum spanning tree distances for each node
    for i in range(distance_matrix.shape[0]):
        # Exclude the current node and calculate the minimum spanning tree
        temp_matrix = np.delete(distance_matrix, i, axis=0)
        temp_matrix = np.delete(temp_matrix, i, axis=1)
        min_spanning_tree = np.min(temp_matrix, axis=1)
        
        # Normalize the minimum spanning tree distances
        normalized_mst = min_spanning_tree / np.sum(min_spanning_tree)
        
        # Update the heuristic matrix for the current node
        for j in range(distance_matrix.shape[0]):
            if i != j:
                heuristic_matrix[i][j] = 1 - normalized_mst[j]
    
    return heuristic_matrix