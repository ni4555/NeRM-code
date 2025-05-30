import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix
    min_distance = np.min(distance_matrix)
    max_distance = np.max(distance_matrix)
    normalized_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    
    # Calculate the minimum spanning tree (MST) to use it for the heuristic
    # Placeholder for MST calculation - This should be implemented using a MST algorithm like Kruskal's or Prim's
    # For demonstration purposes, we will create a matrix that suggests all edges are equally good
    mst_based_heuristic = np.ones_like(normalized_matrix)
    
    # Combine normalized distance with MST-based heuristic
    combined_heuristic = normalized_matrix * mst_based_heuristic
    
    return combined_heuristic